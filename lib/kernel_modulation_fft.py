import os
import h5py
import numpy as np
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap


_lib_dir = os.path.dirname(os.path.abspath(__file__))
h5_path  = '/Users/kennethkohn/Downloads/kernel-9_8um-2umgrid-1umvoxel.h5'
out_dir  = os.path.join(_lib_dir, '../../kernel_results')

plt.style.use(os.path.join(_lib_dir, 'thesisstyle.mpl'))

def custom_cmap():
    '''Wrapper for a custom color map that matches the seismic color map, but has low and high values as the same color. Intended for plotting frequency maps.'''
    colors = [(0, 'blue'), (0.25, 'black'), (0.5, 'red'), (0.75, 'white'), (1, 'blue')]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)

### ── parameters ────────────────────────────────────────────────────────────
### Set to None to use the first available value in the HDF5 file
rbead_select  = None
sep_select    = 10e-6
height_select = 5e-6

### Which harmonic to show in the kernel-vs-r plot (1 = fundamental, 2 = second, …)
harmonic = 1

### V(r) weighting applied to the kernel before integrating over r.
### Set to None for no weighting (uniform V=1).
### Example: V_of_r = lambda r: np.exp(-r / 50e-6)
V_of_r = lambda r:  - G * np.exp(-r / 7.7e-6) / r

y_center     = 0.0e-6      # [m] DC offset of the oscillation
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

### ── time base (shared across all runs) ─────────────────────────────────────
N_t = N_periods * N_per_period
dt  = 1.0 / (freq * N_per_period)
t   = np.arange(N_t) * dt

f_interp = interp1d(yposvec, kernel_slice, axis=0,
                    bounds_error=False, fill_value=0.0)
V        = np.ones(len(r_vals)) if V_of_r is None else V_of_r(r_vals)
freqs    = np.fft.rfftfreq(N_t, d=dt)

def _compute(yc):
    """Run the full pipeline for a given y_center offset."""
    y_t_            = yc + amplitude * np.sin(2.0 * np.pi * freq * t)
    kernel_t_       = f_interp(y_t_)
    weighted_       = kernel_t_ * V[np.newaxis, :, np.newaxis]
    force_t_        = np.sum(weighted_, axis=1) * dr
    fft_t_          = np.fft.rfft(force_t_, axis=0)
    fft_kernel_     = np.fft.rfft(weighted_, axis=0)
    return dict(
        y_t=y_t_, force_t=force_t_,
        fft_t=fft_t_,
        fft_amp=np.abs(fft_t_) * 2.0 / N_t,
        fft_phase=np.angle(fft_t_),
        fft_kernel=fft_kernel_,
    )

### ── y_center sweep ──────────────────────────────────────────────────────────
y_centers = np.arange(0,50e-6, 6.25e-6)
runs      = [_compute(yc) for yc in y_centers]

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

def _plot_run(run, yc):
    fig, axes = plt.subplots(4, 2, figsize=(32, 20))

    y_t_      = run['y_t']
    force_t_  = run['force_t']
    fft_amp_  = run['fft_amp']
    fft_phase_= run['fft_phase']

    axes[0, 0].plot(t * 1e3, y_t_ * 1e6)
    axes[0, 0].set_xlabel('Time [ms]')
    axes[0, 0].set_ylabel('y [μm]')
    axes[0, 0].set_title('y(t) modulation')

    y_fft_raw   = np.fft.rfft(y_t_)
    sc0 = _fft_scatter(axes[0, 1], freqs[1:],
                       np.abs(y_fft_raw)[1:] * 2.0 / N_t,
                       np.angle(y_fft_raw)[1:], f_max)
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('|FFT y| [m]')
    axes[0, 1].set_title('FFT y(t)')
    axes[0, 1].set_xlim(0, f_max)
    cb0 = fig.colorbar(sc0, ax=axes[0, 1], label='Phase [rad]')
    cb0.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb0.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=11)

    for i, lab in enumerate(labels):
        ax = axes[i + 1, 0]
        ax.plot(t * 1e3, force_t_[:, i] - force_t_[0, i])
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(f'F_{lab} [arb]')
        ax.set_title(f'F_{lab}(t)')

        ax = axes[i + 1, 1]
        sc = _fft_scatter(ax, freqs[1:], fft_amp_[:, i][1:], fft_phase_[:, i][1:], f_max)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(f'|FFT F_{lab}|')
        ax.set_title(f'FFT F_{lab}')
        ax.set_xlim(0, f_max)
        cb = fig.colorbar(sc, ax=ax, label='Phase [rad]')
        cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=11)

    fig.suptitle(
        f'Kernel modulation — y₀={yc*1e6:.0f} μm  rbead={rbead*1e6:.2f} μm  '
        f'sep={sep*1e6:.1f} μm  A={amplitude*1e6:.1f} μm  f={freq:.0f} Hz',
        fontsize=11,
    )
    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=6.0)
    return fig

for yc, run in zip(y_centers, runs):
    fig = _plot_run(run, yc)
    fig_path = os.path.join(out_dir, f'kernel_modulation_fft_yc{yc*1e6:.0f}um.png')
    fig.savefig(fig_path, dpi=150)
    print(f'Saved plot  → {fig_path}')
    plt.close(fig)

### ── kernel vs r — interactive harmonic + y_center sliders ──────────────────
from matplotlib.widgets import Slider

max_harmonic = 10

fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6), sharey=False)
fig2.subplots_adjust(bottom=0.32, right=0.90, wspace=0.2)

ax_cb2 = fig2.add_axes([0.95, 0.2, 0.02, 0.70])
cb2 = fig2.colorbar(plt.cm.ScalarMappable(norm=_cnorm, cmap=_cmap), cax=ax_cb2,
                    label='Phase [rad]')
cb2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb2.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=11)

def _harm_idx(n):
    return np.argmin(np.abs(freqs - n * freq))

_state = {'n': harmonic, 'yci': 0}

def _draw(n=None, yci=None):
    if n is not None:
        _state['n'] = int(n)
    if yci is not None:
        _state['yci'] = int(yci)
    n_   = _state['n']
    yci_ = _state['yci']

    hi      = _harm_idx(n_)
    hf      = freqs[hi]
    fk      = runs[yci_]['fft_kernel']
    k_amp   = np.abs(fk[hi]) * 2.0 / N_t
    k_phase = np.angle(fk[hi])
    yc_     = y_centers[yci_]

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
        ax.set_title(f'F_{lab}  —  harmonic {n_}  ({hf:.1f} Hz)', fontsize=18)
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-40, 3e-19)

    fig2.suptitle(
        f'Kernel vs r — harmonic {n_} ({hf:.1f} Hz)  y₀={yc_*1e6:.2f} μm  |  '
        f'rbead={rbead*1e6:.2f} μm  sep={sep*1e6:.1f} μm  A={amplitude*1e6:.1f} μm',
        fontsize=20,
    )
    fig2.canvas.draw_idle()

_draw()

ax_harm_slider = fig2.add_axes([0.15, 0.14, 0.70, 0.04])
slider = Slider(ax_harm_slider, 'Harmonic', 1, max_harmonic,
                valinit=harmonic, valstep=1)
slider.on_changed(lambda v: _draw(n=v))

ax_yc_slider = fig2.add_axes([0.15, 0.06, 0.70, 0.04])
yc_labels = [f'{yc*1e6:.2f} μm' for yc in y_centers]
yc_slider = Slider(ax_yc_slider, 'y₀', 0, len(y_centers) - 1,
                   valinit=0, valstep=1)
yc_slider.valtext.set_text(yc_labels[0])
def _on_yc(v):
    yci = int(v)
    yc_slider.valtext.set_text(yc_labels[yci])
    _draw(yci=yci)
yc_slider.on_changed(_on_yc)

for yci, yc in enumerate(y_centers):
    for n in range(1, max_harmonic + 1):
        _draw(n=n, yci=yci)
        fig2_path = os.path.join(out_dir, f'kernel_vs_r_harmonic{n}_yc{yc*1e6:.0f}um.png')
        fig2.savefig(fig2_path, dpi=150)
        print(f'Saved plot  → {fig2_path}')

_draw(n=harmonic, yci=0)

### ── HTML export (plotly) ────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _plotly_scale = [
        [0.00, 'rgb(0,0,255)'],
        [0.25, 'rgb(0,0,0)'],
        [0.50, 'rgb(255,0,0)'],
        [0.75, 'rgb(255,255,255)'],
        [1.00, 'rgb(0,0,255)'],
    ]

    combos = [
        (yci, n)
        for n in range(1, max_harmonic + 1)
        for yci in range(len(y_centers))
    ]
    n_total = len(combos) * 3

    fig_html = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'F<sub>{l}</sub>' for l in ['x', 'y', 'z']],
        horizontal_spacing=0.10,
    )

    for ci, (yci, n) in enumerate(combos):
        hi      = _harm_idx(n)
        fk      = runs[yci]['fft_kernel']
        k_amp   = np.abs(fk[hi]) * 2.0 / N_t
        k_phase = np.angle(fk[hi])
        yc_     = y_centers[yci]

        for i, lab in enumerate(['x', 'y', 'z']):
            amp_i = k_amp[:, i].copy()
            amp_i[amp_i <= 0] = np.nan
            fig_html.add_trace(
                go.Scatter(
                    x=r_vals * 1e6,
                    y=amp_i,
                    mode='markers',
                    marker=dict(
                        color=k_phase[:, i],
                        colorscale=_plotly_scale,
                        cmin=-np.pi,
                        cmax=np.pi,
                        size=4,
                        showscale=(i == 2),
                        colorbar=dict(
                            title='Phase [rad]',
                            tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                            ticktext=['-π', '-π/2', '0', 'π/2', 'π'],
                            x=1.03, thickness=15, len=0.8,
                        ) if i == 2 else {},
                    ),
                    visible=(ci == 0),
                    showlegend=False,
                    name=f'n={n}, y₀={yc_*1e6:.2f}μm, F{lab}',
                ),
                row=1, col=i + 1,
            )

    slider_steps = []
    for ci, (yci, n) in enumerate(combos):
        vis = [False] * n_total
        for j in range(3):
            vis[ci * 3 + j] = True
        yc_ = y_centers[yci]
        slider_steps.append({
            'args': [{'visible': vis}],
            'label': f'n={n} | y₀={yc_*1e6:.2f}μm',
            'method': 'restyle',
        })

    fig_html.update_layout(
        height=520,
        margin=dict(b=130, t=80, r=100),
        plot_bgcolor='#808080',
        paper_bgcolor='white',
        title=dict(
            text=(
                f'Kernel vs r — rbead={rbead*1e6:.2f} μm  '
                f'sep={sep*1e6:.1f} μm  A={amplitude*1e6:.1f} μm'
            ),
            font=dict(size=15),
        ),
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(prefix='', xanchor='center', font=dict(size=13)),
            pad=dict(t=50),
            len=0.92,
            x=0.04,
        )],
    )

    for col in range(1, 4):
        yax = 'yaxis' if col == 1 else f'yaxis{col}'
        xax = 'xaxis' if col == 1 else f'xaxis{col}'
        lab = ['x', 'y', 'z'][col - 1]
        fig_html.update_layout({
            yax: dict(type='log', range=[-40, -18],
                      title=f'|FFT kernel F<sub>{lab}</sub>| [arb]'),
            xax: dict(range=[0, 100], title='r [μm]'),
        })

    html_path = os.path.join(out_dir, 'kernel_vs_r_interactive.html')
    fig_html.write_html(html_path, include_plotlyjs='cdn')
    print(f'Saved HTML  → {html_path}')

except ImportError:
    print('plotly not installed — skipping HTML export (pip install plotly)')

plt.show()
