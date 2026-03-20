"""
Wendling Neural Mass Model Simulator (SEEG Generation)

This script simulates macroscopic depth-EEG (SEEG) signals using the 
computational neural mass model of excitatory and inhibitory populations.
It models the transition from healthy background activity to epileptic seizures
(ictal state) by altering the balance of slow dendritic and fast somatic
GABAergic inhibition.

Methodology:
- Implements the 10-dimensional stochastic differential equation (SDE) system.
- Uses a fixed-setp Euler-Maruyama integration method running at 10 kHz to 
    prevent numerical damping of fast physiological frequencies (Gamma band).
- Outputs realistic SEEG traces and parameter evolution spectrograms. 

Reference:
    Wending, F., Bartolomei, F., Bellanger, J. J., & Chauvel, P. (2002).
    "Epileptic fast activity can be explained by a model of impaired
    GABAergic dendritic inhibition." European Journal of Neuroscience, 15(9)

Author: [Jerem-loq]
Date: [March 2026]
Licence: [Open Source]

Dependencies:
- numpy     1.26.4
- scipy     1.12.0
- matplolib 3.8.0
- tqdm      4.66.5
"""


import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram
import matplotlib.pyplot as plt
from tqdm import tqdm


def S(v):
    """
    Sigmoid transformation: membrane potential to firing rate.
    """
    e0 = 2.5  # max firing rate | s-1
    v0 = 6.0  # half-activation threshold | mV
    r = 0.56  # steepness (gain) | mV-1
    return (2.0 * e0) / (1.0 + np.exp(r * (v0 - v)))


def simulate_wendling(dur, fs_out=1000, fs_sim=10000):
    """
    Solves the Wendling 4-population model using the Euler-Maruyama method.
    Runs at 10kHz internally to prevent numerical damping of the fast 80Hz dynamics.
    """
    dt = 1.0 / fs_sim
    n_steps_sim = int(dur * fs_sim)
    n_steps_out = int(dur * fs_out)
    ds_factor = fs_sim // fs_out

    t_sim = np.linspace(0, dur, n_steps_sim)
    t_out = np.linspace(0, dur, n_steps_out)
    # t_arr = np.linspace(0, dur, n_steps)

    # 1. Parameter Interpolation (Single source of Truth)
    # Average excitatory synaptic gain
    A_arr = np.full(n_steps_sim, 5.0)  # Constant

    """
    If non-linearity:
    1. 00% to 10%   : Hold Healthy (B=40, G=20)
    2. 10% to 35%   : Types 2 & 3 - Spikes (drop B to ~22)
    3. 35% to 55%   : Type 4 - Slow rhythmic (drop B to ~15)
    4. 55% to 75%   : Type 5 - Fast gamma activity (drop B to ~5)
    5. 75% to 85%   : Hold fast activity
    6. 85% to 100%  : Type 6 - Slow quasi-sinusoidal (drop G to 5)
    """

    # key_fractions = np.array([0.0, 0.10, 0.35, 0.55, 0.75, 0.85, 1.0])
    # key_times = key_fractions * dur
    key_times = np.array([
        0.0,    # Burn-in start
        100.0,   # End of plateau 1 (Healthy) 25
        103.0,   # Rapid ramp B down 28
        130.0,   # End of plateau 2 (Spikes) 55
        133.0,   # Rapid ramp B down  58 
        150.0,   # End of plateau 3 (Slow rhythmic) 85
        175.0,   # Rapid ramp B down 88
        200.0,  # End of plateau 4 (Fast gamma activity) 125
        203.0,  # Rapid ramp G down 128
        235.0   # End of plateau 5 (Slow quasi-sinusoidal) 160
    ])

    # Average slow inhibitory synaptic gain (Healthy to Ictal)
    # E.g., drops from 22.0 to 2.0
    # B_arr = np.linspace(45.0, 2.0, n_steps_sim)  # Linear
    B_keys = np.array([40.0, 40.0, 22.0, 22.0, 15.0, 15.0, 5.0, 5.0, 10.0, 10.0])
    B_arr = np.interp(t_sim, key_times, B_keys)

    # Average fast inhibitory synaptic gain (Healthy to Ictal)
    # E.g., increases from 10.0 to 20.0
    # G_arr = np.linspace(5.0, 25.0, n_steps_sim)
    # G_arr = np.full(n_steps_sim, 20)
    G_keys = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 5.0, 5.0])
    G_arr = np.interp(t_sim, key_times, G_keys)


    # Pre-generate Gaussian white noise p(t)
    p_t_1k = np.random.normal(90, 30, n_steps_out)  # Generate noise at 1 kHz to maintain the correct physiological power spectrum
    p_t = np.repeat(p_t_1k, ds_factor)  # Hold each noise value for ds_factor steps (10) for the 10 kHz simulation

    # Constants from table 1
    a, b, g = 100.0, 50.0, 500.0  # s-1
    C = 135.0
    C1, C2, C3, C4, C5, C6, C7 = C, 0.8*C, 0.25*C, 0.25*C, 0.3*C, 0.1*C, 0.8*C

    # Initialize state variables (y0 to y9)
    y = np.zeros(10)

    # Pre-allocate raw EEG array
    eeg_out = np.zeros(n_steps_out)

    print(f"Simulating Wendling Model (Euler Method - fs={fs_sim}Hz) for {dur}s...")

    # Fixed-step integration loop
    out_idx = 0
    for i in tqdm(range(n_steps_sim), desc="Solving SDE"):
        A = A_arr[i]
        B = B_arr[i]
        G = G_arr[i]
        p = p_t[i]

        # Unpack current state
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9 = y

        # Calculate derivatives
        dy0 = y5
        dy5 = A * a * S(y1 - y2 - y3) - 2 * a * y5 - a**2 * y0

        dy1 = y6
        dy6 = A * a * (p + C2 * S(C1 * y0)) - 2 * a * y6 - a**2 * y1

        dy2 = y7
        dy7 = B * b * C4 * S(C3 * y0) - 2 * b * y7 - b**2 * y2

        dy3 = y8
        dy8 = G * g * C7 * S(C5 * y0 - C6 * y4) - 2 * g * y8 - g**2 * y3

        dy4 = y9
        dy9 = B * b * S(C3 * y0) - 2 * b * y9 - b**2 * y4

        # Apply Euler step (y_next = y_current + dy * dt)
        y[0] += dy0 * dt
        y[1] += dy1 * dt
        y[2] += dy2 * dt
        y[3] += dy3 * dt
        y[4] += dy4 * dt
        y[5] += dy5 * dt
        y[6] += dy6 * dt
        y[7] += dy7 * dt
        y[8] += dy8 * dt
        y[9] += dy9 * dt

        # Downsample and save to output array at 1,000 Hz
        if i % ds_factor == 0 and out_idx < n_steps_out:
            eeg_out[out_idx] = y[1] - y[2] - y[3]
            out_idx += 1

        # # Record output: summation of PSPs at the pyramidal level
        # eeg_raw[i] = y[1] - y[2] - y[3]

    return t_out, eeg_out, A_arr[::ds_factor], B_arr[::ds_factor], G_arr[::ds_factor]


if __name__ == "__main__":
    fs = 1_000
    dur = 260.0

    # Run the custom solver
    t, eeg_raw, A_arr, B_arr, G_arr = simulate_wendling(dur, fs_out=fs)

    # Remove burn-in (first 10s)
    cut_off = 10 * fs
    eeg = eeg_raw[cut_off:]
    t_stable = t[cut_off:] - 10  # Shift time to start at 0
    A_stable = A_arr[cut_off:]
    B_stable = B_arr[cut_off:]
    G_stable = G_arr[cut_off:]

    # High-pass filtering
    sos = butter(4, 0.5, btype='high', fs=fs, output='sos')
    eeg = sosfiltfilt(sos, eeg)

    # Scaling
    ambient_noise = np.random.normal(0, 0.3, len(eeg))
    eeg += ambient_noise
    eeg *= 10.0

    version = "test_6"
    np.save(f"simm_eeg_trace_NL_{version}.npy", eeg.astype(np.float64))
    np.savez(f"sim_eeg_param_NL_{version}.npz", time=t_stable, A=A_stable, B=B_stable, G=G_stable)
    print(f"Saved 'sim_eeg_trace{version}.npy' and its parameters (.npz).")

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 2, 1]})

    # 1. Raw trace
    ax1.plot(t_stable, eeg, color='black', linewidth=0.8)
    ax1.set_title("Simulated EEG: Healthy to Ictal transition")
    ax1.set_ylabel("Amplitude (µV)")

    # 2. Spectrogram
    f_spec, t_spec, Sxx = spectrogram(eeg, fs, nperseg=1024, noverlap=512)
    freq_mask = f_spec <= 150  # Limit to 150 Hz to see gamma clearly
    im = ax2.pcolormesh(t_spec + t_stable[0], f_spec[freq_mask],
                        10 * np.log10(Sxx[freq_mask, :]),
                        shading='gouraud', cmap='jet')
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Spectrogram (Power in dB)")
    # fig.colorbar(im, ax=ax2, label='Power(dB)')

    # 3. Parameters
    ax3.plot(t_stable, B_stable, label="B (Slow dendritic Inh.)", color='green')
    ax3.plot(t_stable, G_stable, label="G (Fast somatic Inh.)", color='purple')
    ax3.plot(t_stable, A_stable, label="A (Excitation.)", color='red')
    ax3.set_title("Parameter Evolution")
    ax3.set_ylabel("Gain Value")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper right', ncol=3)

    plt.tight_layout()
    plt.show()

    print(" ")
