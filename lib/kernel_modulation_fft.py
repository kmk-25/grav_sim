import os
import h5py
import numpy as np
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap


_lib_dir = os.path.dirname(os.path.abspath(__file__))
h5_path  = os.path.join(_lib_dir, '../../kernel_results/kernel_table.h5')
out_dir  = os.path.join(_lib_dir, '../../kernel_results')

plt.style.use(os.path.join(_lib_dir, 'thesisstyle.mpl'))

def custom_cmap():
    '''Wrapper for a custom color map that matches the seismic color map, but has low and high values as the same color. Intended for plotting frequency maps.'''
    colors = [(0, 'blue'), (0.25, 'black'), (0.5, 'red'), (0.75, 'white'), (1, 'blue')]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)

### ── parameters ────────────────────────────────────────────────────────────
### Set to None to use the first available value in the HDF5 file
rbead_select  = None
sep_select    = None
height_select = None

### Which harmonic to show in the kernel-vs-r plot (1 = fundamental, 2 = second, …)
harmonic = 1

### V(r) weighting applied to the kernel before integrating over r.
### Set to None for no weighting (uniform V=1).
### Example: V_of_r = lambda r: np.exp(-r / 50e-6)
V_of_r = lambda r:  - G * np.exp(-r / 7.7e-6) / r

y_center     = 0.0       # [m] DC offset of the oscillation
amplitude    = - 85.0e-6   # [m] oscillation amplitude (~1 finger pitch)
freq         = 3.0      # [Hz] modulation frequency
N_periods    = 1        # number of periods to simulate
N_per_period = 1000      # time samples per period (sets max observable harmonic)

### ── load ───────────────────────────────────────────────────────────────────
with h5py.File(h5_path, 'r') as hf:
    rbeads  = hf['rbeads'][:]
    seps    = hf['seps'][:]
    heights = hf['heights'][:]
    r_vals  = hf['r_vals'][:]
    yposvec = hf['yposvec'][:]

    i_rb  = 0 if rbead_select  is None else np.argmin(np.abs(rbeads  - rbead_select))
    i_sep = 0 if sep_select    is None else np.argmin(np.abs(seps    - sep_select))
    i_h   = 0 if height_select is None else np.argmin(np.abs(heights - height_select))

    kernel_slice = hf['kernel'][i_rb, i_sep, i_h]  # (N_ypos, N_r, 3)
    rho_bead     = hf.attrs['rho_bead']

rbead  = rbeads[i_rb]
sep    = seps[i_sep]
height = heights[i_h]
dr     = r_vals[1] - r_vals[0]

print(f'rbead={rbead*1e6:.3f} um  sep={sep*1e6:.1f} um  height={height*1e6:.1f} um')
print(f'kernel_slice shape: {kernel_slice.shape}  (N_ypos, N_r, 3)')

### ── time vector and y(t) ───────────────────────────────────────────────────
N_t = N_periods * N_per_period
dt  = 1.0 / (freq * N_per_period)
t   = np.arange(N_t) * dt
y_t = y_center + amplitude * np.sin(2.0 * np.pi * freq * t)

### ── interpolate kernel at each y(t), integrate over r ─────────────────────
# interp1d over ypos axis; fill with 0 outside the sampled range
f_interp = interp1d(yposvec, kernel_slice, axis=0,
                    bounds_error=False, fill_value=0.0)
kernel_t        = f_interp(y_t)                                         # (N_t, N_r, 3)
V               = np.ones(len(r_vals)) if V_of_r is None else V_of_r(r_vals)
weighted_kernel = kernel_t * V[np.newaxis, :, np.newaxis]               # (N_t, N_r, 3)
force_t         = np.sum(weighted_kernel, axis=1) * dr                  # (N_t, 3)

### ── FFT ────────────────────────────────────────────────────────────────────
freqs      = np.fft.rfftfreq(N_t, d=dt)                                 # (N_freq,)
fft_t      = np.fft.rfft(force_t, axis=0) * np.exp(-1j*np.pi/2)                               # (N_freq, 3)
fft_amp    = np.abs(fft_t) * 2.0 / N_t

fft_kernel = np.fft.rfft(weighted_kernel, axis=0)                       # (N_freq, N_r, 3)

### ── colormap ────────────────────────────────────────────────────────────────
_cmap  = custom_cmap()
_cnorm = plt.Normalize(vmin=-np.pi, vmax=np.pi)

### ── scatter-plot helper ────────────────────────────────────────────────────
scatter_interval = freq   # [Hz] plot one point every this many Hz

_bg = '#808080'

def _fft_scatter(ax, freqs, amp, phase, f_max):
    """Line plot of FFT amplitude with a phase-coloured scatter overlay at every scatter_interval Hz."""
    ax.set_facecolor(_bg)
    mask = (np.abs(freqs % scatter_interval) < (freqs[1] - freqs[0]) * 0.5) & (freqs <= f_max)
    ax.plot(freqs[freqs <= f_max], amp[freqs <= f_max],
            color='white', linewidth=0.8, zorder=2)
    sc = ax.scatter(freqs[mask], amp[mask],
                    c=phase[mask], cmap=_cmap, norm=_cnorm,
                    s=40, zorder=3)
    # ax.set_yscale('log')
    return sc

### ── plot ───────────────────────────────────────────────────────────────────
labels = ['x', 'y', 'z']
n_harm = 10
f_max  = n_harm * freq

fft_phase = np.angle(fft_t)   # (N_freq, 3),  range −π … +π

fig, axes = plt.subplots(4, 2, figsize=(32, 20))

# row 0: y(t) and its FFT (scatter)
axes[0, 0].plot(t * 1e3, y_t * 1e6)
axes[0, 0].set_xlabel('Time [ms]')
axes[0, 0].set_ylabel('y [μm]')
axes[0, 0].set_title('y(t) modulation')

y_fft_raw   = np.fft.rfft(y_t)
y_fft_amp   = np.abs(y_fft_raw) * 2.0 / N_t
y_fft_phase = np.angle(y_fft_raw)
sc0 = _fft_scatter(axes[0, 1], freqs, y_fft_amp, y_fft_phase, f_max)
axes[0, 1].set_xlabel('Frequency [Hz]')
axes[0, 1].set_ylabel('|FFT y| [m]')
axes[0, 1].set_title('FFT y(t)')
axes[0, 1].set_xlim(0, f_max)
cb0 = fig.colorbar(sc0, ax=axes[0, 1], label='Phase [rad]')
cb0.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb0.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=11)

# rows 1-3: force components
for i, lab in enumerate(labels):
    # time domain
    ax = axes[i + 1, 0]
    ax.plot(t * 1e3, force_t[:, i] - force_t[0,i])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(f'F_{lab} [arb]')
    ax.set_title(f'F_{lab}(t)')

    # frequency domain — scatter coloured by phase
    ax = axes[i + 1, 1]
    sc = _fft_scatter(ax, freqs, fft_amp[:, i], fft_phase[:, i], f_max)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'|FFT F_{lab}|')
    ax.set_title(f'FFT F_{lab}')
    ax.set_xlim(0, f_max)
    cb = fig.colorbar(sc, ax=ax, label='Phase [rad]')
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=11)

fig.suptitle(
    f'Kernel modulation — rbead={rbead*1e6:.2f} μm  sep={sep*1e6:.1f} μm  '
    f'A={amplitude*1e6:.1f} μm  f={freq:.0f} Hz',
    fontsize=11,
)
fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=6.0)

fig_path = os.path.join(out_dir, 'kernel_modulation_fft.png')
fig.savefig(fig_path, dpi=150)
print(f'Saved plot  → {fig_path}')

### ── save data ──────────────────────────────────────────────────────────────
npz_path = os.path.join(out_dir, 'kernel_modulation_fft.npz')
np.savez(npz_path,
    t=t, y_t=y_t,
    force_t=force_t,
    freqs=freqs, fft_amp=fft_amp,
    rbead=np.array(rbead), sep=np.array(sep), height=np.array(height),
    amplitude=np.array(amplitude), freq=np.array(freq),
)
print(f'Saved data  → {npz_path}')

### ── kernel vs r — interactive harmonic slider ───────────────────────────────
from matplotlib.widgets import Slider

max_harmonic = int(freqs[-1] / freq)

fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6), sharey=False)
fig2.subplots_adjust(bottom=0.28, right=0.90, wspace=0.2)

ax_cb2 = fig2.add_axes([0.95, 0.2, 0.02, 0.70])
cb2 = fig2.colorbar(plt.cm.ScalarMappable(norm=_cnorm, cmap=_cmap), cax=ax_cb2,
                    label='Phase [rad]')
cb2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb2.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=11)

def _harm_idx(n):
    return np.argmin(np.abs(freqs - n * freq))

def _draw(n):
    hi       = _harm_idx(n)
    hf       = freqs[hi]
    k_amp    = np.abs(fft_kernel[hi]) * 2.0 / N_t
    k_phase  = np.angle(fft_kernel[hi])
    for i, lab in enumerate(labels):
        ax = axes2[i]
        ax.cla()
        ax.set_facecolor(_bg)
        ax.scatter(r_vals * 1e6, k_amp[:, i],
                   c=k_phase[:, i], cmap=_cmap, norm=_cnorm,
                   s=30, zorder=3)
        ax.set_yscale('log')
        ax.set_xlabel('r [μm]', fontsize=16)
        ax.set_ylabel(f'|FFT kernel F_{lab}| [arb]', fontsize=16)
        ax.tick_params(labelsize=11)
        ax.set_title(f'F_{lab}  —  harmonic {n}  ({hf:.1f} Hz)', fontsize=18)
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-40,3e-19)

    fig2.suptitle(
        f'Kernel vs r — harmonic {n} ({hf:.1f} Hz)  |  '
        f'rbead={rbead*1e6:.2f} μm  sep={sep*1e6:.1f} μm  A={amplitude*1e6:.1f} μm',
        fontsize=20,
    )
    fig2.canvas.draw_idle()

_draw(harmonic)

ax_slider = fig2.add_axes([0.15, 0.05, 0.70, 0.04])
slider = Slider(ax_slider, 'Harmonic', 1, max_harmonic,
                valinit=harmonic, valstep=1)
slider.on_changed(_draw)

for n in range(1, 11):
    _draw(n)
    fig2_path = os.path.join(out_dir, f'kernel_vs_r_harmonic{n}.png')
    fig2.savefig(fig2_path, dpi=150)
    print(f'Saved plot  → {fig2_path}')

_draw(harmonic)
plt.show()
