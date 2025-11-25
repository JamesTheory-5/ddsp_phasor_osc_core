# ddsp_phasor_osc_core

MODULE NAME:
**ddsp_phasor_osc_core**

DESCRIPTION:
A fully differentiable, pure-JAX **phasor oscillator** in GDSP style.
`ddsp_phasor_osc_core` integrates frequency in Hz into a wrapped, normalized phase signal suitable for driving other oscillators (sine, saw, square, wavetable, FM/PM, etc.). It includes optional one-pole frequency smoothing and supports centered or uncentered phase outputs.

---

INPUTS:

* **x (freq_hz)** : input frequency in Hz at the current sample (scalar in `tick`, array in `process`)
* **params.dt** : time step in seconds (`1 / sample_rate`), scalar
* **params.smooth_alpha** : one-pole smoothing coefficient α ∈ [0,1] for frequency
* **params.phase_offset** : normalized phase offset added after integration (0–1 range)
* **params.centered_flag** : scalar 0.0 or 1.0; 0.0 → output in `[0,1)`, 1.0 → output in `[-0.5,0.5)`

OUTPUTS:

* **y (phase_out)** : current phase sample, wrapped and optionally centered
* **new_state** : updated state tuple

---

STATE VARIABLES:

```python
(
    phase,        # current phase in [0,1)
    freq_smooth,  # smoothed frequency in Hz
)
```

Both `phase` and `freq_smooth` are JAX arrays/scalars, no Python floats inside jit.

---

EQUATIONS / MATH:

Let:

* `x[n]` = input frequency in Hz at sample n
* `dt` = time step (1 / sample_rate)
* `α` = smoothing coefficient `smooth_alpha`
* `phase[n]` = current stored phase in [0,1)
* `phase_offset` = normalized phase offset
* `centered_flag` ∈ {0,1}

Frequency smoothing:

```text
freq_smooth[n+1] = freq_smooth[n] + α * (x[n] − freq_smooth[n])
```

Phase integration (through-zero capable, since freq can be negative):

```text
phase_unwrapped[n+1] = phase[n] + freq_smooth[n+1] * dt
phase_wrapped[n+1]   = phase_unwrapped[n+1] mod 1
```

Phase offset and optional centering:

```text
phase_shifted[n+1] = phase_wrapped[n+1] + phase_offset
phase_shifted[n+1] = phase_shifted[n+1] − floor(phase_shifted[n+1])    # [0,1)

phase_centered[n+1] = phase_shifted[n+1] − 0.5    # [-0.5, 0.5)

y[n] = phase_shifted[n+1] * (1 − centered_flag) + phase_centered[n+1] * centered_flag
```

State update:

```text
phase[n+1]      = phase_wrapped[n+1]
freq_smooth[n+1] as above
```

through-zero rules:
Through-zero operation is automatic; negative frequency values cause backward phase motion, which is then wrapped by the modulus operation.

phase wrapping rules:
All wrapping is done via `jnp.mod` or `x - jnp.floor(x)` to keep phase in `[0,1)`.

nonlinearities:
Only linear operations plus modulo and floor; everything is differentiable with respect to frequency and smoothing parameters, except at the wrap discontinuity (as expected for a saw/phase ramp).

interpolation rules:
No interpolation here; this module only provides phase. Interpolation of tables or delay lines is delegated to `interp_core` or `table_core`.

any time-varying coefficient rules:

* `smooth_alpha`, `phase_offset`, and `centered_flag` are typically scalars but may be passed as per-sample arrays (broadcasted in `process`).
* Frequency input `x` is fully time-varying.

---

NOTES:

* Stable ranges:

  * `smooth_alpha ∈ [0,1]` (clipped inside jit).
  * `dt > 0`.
* Phase is always stored internally in `[0,1)`, regardless of centered output.
* No arrays are allocated inside jit; all buffers are created outside jit.
* No Python branching inside jit; all control flow is via arithmetic and JAX primitives.

---

## Full `ddsp_phasor_osc_core.py`

```python
"""
ddsp_phasor_osc_core.py

GammaJAX DDSP – Phasor Oscillator Core
--------------------------------------

This module implements a fully differentiable, pure-JAX phasor oscillator in
GDSP style:

    ddsp_phasor_osc_core_init(...)
    ddsp_phasor_osc_core_update_state(...)
    ddsp_phasor_osc_core_tick(freq_hz, state, params)
    ddsp_phasor_osc_core_process(freq_hz_buf, state, params)

The phasor oscillator integrates frequency in Hz into a normalized phase signal
in [0,1), with optional phase centering and one-pole frequency smoothing.

Design constraints:
    - Pure functional JAX.
    - No classes, no dicts, no dataclasses.
    - State = tuple only (arrays/scalars).
    - tick() returns (y, new_state).
    - process() is a lax.scan wrapper around tick().
    - No Python branching inside @jax.jit.
    - No dynamic allocation or jnp.arange/jnp.zeros inside jit.
    - All shapes determined outside jit.
    - All control flow via jnp.where / lax.cond / lax.scan.
    - Everything jittable and differentiable.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# 1. GDSP-style API: init / update_state / tick / process
# =============================================================================

def ddsp_phasor_osc_core_init(
    initial_phase: float,
    initial_freq_hz: float,
    sample_rate: float,
    smooth_alpha: float = 0.0,
    phase_offset: float = 0.0,
    centered: bool = False,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Initialize phasor oscillator core.

    Args:
        initial_phase  : starting phase in [0,1)
        initial_freq_hz: initial frequency in Hz
        sample_rate    : sample rate in Hz (used to compute dt = 1 / sample_rate)
        smooth_alpha   : one-pole smoothing coefficient for frequency in [0,1]
                         0  => no smoothing (freq_smooth follows freq exactly)
                         1  => heavy smoothing / slow response
        phase_offset   : normalized phase offset (0..1) applied after wrap
        centered       : if True, output phase in [-0.5,0.5) instead of [0,1)
        dtype          : JAX dtype for internal state

    Returns:
        state  : (phase, freq_smooth)
        params : (dt, smooth_alpha, phase_offset, centered_flag)
    """
    phase0 = jnp.asarray(initial_phase, dtype=dtype)
    # Wrap initial phase into [0,1)
    phase0 = jnp.mod(phase0, 1.0)

    freq0 = jnp.asarray(initial_freq_hz, dtype=dtype)
    dt = jnp.asarray(1.0 / float(sample_rate), dtype=dtype)
    alpha = jnp.asarray(smooth_alpha, dtype=dtype)

    phase_offset_arr = jnp.asarray(phase_offset, dtype=dtype)
    centered_flag = jnp.asarray(1.0 if centered else 0.0, dtype=dtype)

    state = (phase0, freq0)
    params = (dt, alpha, phase_offset_arr, centered_flag)
    return state, params


def ddsp_phasor_osc_core_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optional non-IO state update.

    For this simple phasor, smoothing is applied inside tick(),
    so update_state is currently a pass-through.

    Args:
        state  : (phase, freq_smooth)
        params : (dt, smooth_alpha, phase_offset, centered_flag)  # unused here

    Returns:
        new_state: (phase, freq_smooth)
    """
    del params
    return state


@jax.jit
def ddsp_phasor_osc_core_tick(
    freq_hz: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Single-sample phasor oscillator tick.

    Inputs:
        freq_hz : input frequency in Hz (scalar)
        state   : (phase, freq_smooth)
        params  : (dt, smooth_alpha, phase_offset, centered_flag)

    Returns:
        y          : phase sample (either in [0,1) or [-0.5,0.5))
        new_state  : (phase_next, freq_smooth_next)
    """
    phase, freq_smooth = state
    dt, alpha, phase_offset, centered_flag = params

    # Cast everything consistently
    freq_hz = jnp.asarray(freq_hz, dtype=phase.dtype)
    dt = jnp.asarray(dt, dtype=phase.dtype)
    alpha = jnp.asarray(alpha, dtype=phase.dtype)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    phase_offset = jnp.asarray(phase_offset, dtype=phase.dtype)
    centered_flag = jnp.asarray(centered_flag, dtype=phase.dtype)

    # One-pole smoothing of frequency:
    # freq_smooth_next = freq_smooth + alpha * (freq_hz - freq_smooth)
    freq_smooth_next = freq_smooth + alpha * (freq_hz - freq_smooth)

    # Phase update (through-zero capable)
    phase_unwrapped = phase + freq_smooth_next * dt
    phase_wrapped = jnp.mod(phase_unwrapped, 1.0)

    # Phase offset, wrapped into [0,1)
    phase_shifted = phase_wrapped + phase_offset
    phase_shifted = phase_shifted - jnp.floor(phase_shifted)

    # Optional centered phase in [-0.5,0.5)
    phase_centered = phase_shifted - 0.5
    y = phase_shifted * (1.0 - centered_flag) + phase_centered * centered_flag

    new_state = (phase_wrapped, freq_smooth_next)
    return y, new_state


@jax.jit
def ddsp_phasor_osc_core_process(
    freq_hz_buf: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer of frequencies into a buffer of phases via lax.scan.

    Args:
        freq_hz_buf : array of input frequencies in Hz, shape (T,)
        state       : (phase, freq_smooth)
        params      : (dt, smooth_alpha, phase_offset, centered_flag)

    Returns:
        phase_buf    : buffer of phases, shape (T,)
        final_state
    """
    freq_hz_buf = jnp.asarray(freq_hz_buf)

    def body(carry, freq_t):
        st = carry
        y_t, st_next = ddsp_phasor_osc_core_tick(freq_t, st, params)
        return st_next, y_t

    final_state, phase_buf = lax.scan(body, state, freq_hz_buf)
    return phase_buf, final_state


# =============================================================================
# 2. Smoke test, plot, listen
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    print("=== ddsp_phasor_osc_core: smoke test ===")

    # Parameters
    sample_rate = 48000.0
    duration_sec = 0.01  # short for plotting
    N = int(sample_rate * duration_sec)

    # Constant 440 Hz for the smoke test
    freq = 440.0
    freq_buf = jnp.full((N,), freq, dtype=jnp.float32)

    # Initialize phasor oscillator
    state, params = ddsp_phasor_osc_core_init(
        initial_phase=0.0,
        initial_freq_hz=freq,
        sample_rate=sample_rate,
        smooth_alpha=0.05,
        phase_offset=0.0,
        centered=False,
        dtype=jnp.float32,
    )

    # Process
    phase_buf, state_out = ddsp_phasor_osc_core_process(freq_buf, state, params)
    phase_np = onp.asarray(phase_buf)

    # Plot phase ramp
    plt.figure(figsize=(10, 4))
    plt.plot(phase_np, label="phase (0..1)")
    plt.title("ddsp_phasor_osc_core: phase ramp at 440 Hz")
    plt.xlabel("Sample")
    plt.ylabel("Phase")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen example: convert phase to sine
    duration_sec_audio = 1.0
    N_audio = int(sample_rate * duration_sec_audio)
    freq_buf_audio = jnp.full((N_audio,), freq, dtype=jnp.float32)

    phase_audio, _ = ddsp_phasor_osc_core_process(freq_buf_audio, state, params)
    two_pi = 2.0 * jnp.pi
    audio = jnp.sin(two_pi * phase_audio)
    audio_np = onp.asarray(audio) * 0.2  # scale down

    if HAVE_SD:
        print("Playing test sine generated from phasor_osc_core...")
        sd.play(audio_np, samplerate=int(sample_rate), blocking=True)
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

If you’d like, the obvious next step is a **stateless phase → sine module**:

> **“Generate ddsp_sine_osc_from_phase_core.py”**

or a **BLEP-based saw/square** that consumes this phasor phase.
