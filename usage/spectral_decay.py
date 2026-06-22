"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Spectral decay of exp(-c sin(x)) via FFT -- numpy vs mpmath.

Samples f(x) = exp(-c sin(x)) on a uniform periodic grid, computes
FFT coefficients with both numpy.fft (double precision) and
mpmath_fft (arbitrary precision), and plots |c_k| vs k on semilog
axes to show how double precision hits a ~1e-15 noise floor while
mpmath continues to resolve the true exponential decay.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpmath as mp
import numpy as np
from mpmath_fft import fft

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N = 256                      # number of sample points
C = 2.0                      # f(x) = exp(-C * sin(x))
MP_DPS = 45                  # mpmath precision (~1e-40 floor)

# ---------------------------------------------------------------------------
# Sample function
# ---------------------------------------------------------------------------

def func_samples(N, c, dtype=complex):
  """Return f(x_j) for x_j = 2*pi*j/N, j = 0..N-1."""
  x = 2.0 * np.pi * np.arange(N) / N
  if dtype is complex:
    return np.exp(-c * np.sin(x)).astype(complex)
  else:
    # mpmath object array
    out = np.empty(N, dtype=object)
    for j in range(N):
      xj = mp.mpf(2) * mp.pi * j / N
      val = mp.e ** (-mp.mpf(c) * mp.sin(xj))
      out[j] = mp.mpc(val, 0)
    return out


# ---------------------------------------------------------------------------
# Compute and extract magnitude spectrum
# ---------------------------------------------------------------------------

# Numpy FFT
x_np = func_samples(N, C, dtype=complex)
y_np = np.fft.fft(x_np)
coeff_np = np.abs(y_np[:N // 2])
# Clip exact zeros to a visible floor for semilogy
coeff_np = np.maximum(coeff_np, 1e-17)

# mpmath FFT (high precision)
mp.mp.dps = MP_DPS
x_mp = func_samples(N, C, dtype=object)
y_mp = fft(x_mp)
coeff_mp = np.array([float(abs(y_mp[k])) for k in range(N // 2)])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

k = np.arange(N // 2)

ax.semilogy(k, coeff_np, 'o', color='#1f77b4', markersize=3,
            label='numpy.fft (float64)')
ax.semilogy(k, coeff_mp, '-', color='#d62728', linewidth=1.5,
            label='mpmath_fft (dps=%d)' % MP_DPS)

# Noise floor annotation
ax.axhline(1e-16, color='#1f77b4', linestyle=':', alpha=0.5)
ax.text(N // 2 - 5, 1.5e-16, 'float64 floor', color='#1f77b4',
        fontsize=8, ha='right', va='bottom', alpha=0.7)

ax.set_xlabel('wavenumber k', fontsize=12)
ax.set_ylabel(r'$|c_k|$', fontsize=12)
ax.set_title(r'FFT coefficient decay: $f(x) = e^{-C\,\sin x}$'
             '  (C=%g, N=%d)' % (C, N), fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_xlim(0, N // 2 - 1)

fig.tight_layout()
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figs_dir = os.path.join(repo_root, 'figs')
os.makedirs(figs_dir, exist_ok=True)
fig.savefig(os.path.join(figs_dir, 'spectral_decay.png'), dpi=150)
plt.close(fig)

print('Plot saved to figs/spectral_decay.png')
print()
print('Coefficient decay summary:')
print('  k=0   numpy=%.6e  mp=%.6e' % (coeff_np[0], coeff_mp[0]))
for km in [N//8, N//4, N//2 - 1]:
  print('  k=%-4d numpy=%.2e  mp=%.2e  ratio=%.1f'
        % (km, coeff_np[km], coeff_mp[km],
           coeff_np[km] / coeff_mp[km] if coeff_mp[km] > 0 else 0))
