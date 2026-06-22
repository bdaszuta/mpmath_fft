"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Arbitrary-precision FFT kernels for mpmath.

Twiddle factor construction, plan builder, leaf DFT kernels,
Cooley-Tukey helpers, stack-based plan executor.  All arrays
use numpy dtype=object storing mpmath mpc values.

Refs:
  [1] Cooley & Tukey (1965), Math. Comp. 19(90):297-301.
  [2] Bluestein (1970), IEEE Trans. Audio Electroacoustics 18(4):451-455.
  [3] Chu & George (1999), "Inside the FFT Black Box", CRC Press.
  [4] Johnson & Frigo (2009), "Implementing FFTs in Practice."
  [5] Frigo & Johnson (2005), Proc. IEEE 93(2):216-231.
"""
import mpmath as mp
import numpy as np


###############################################################################
# Number theory helpers (self-contained, no pynalgo dependency)
###############################################################################

def _get_prime_factors(n):
  """
  Return prime factorization: list of (prime, power) tuples.
  """
  factors = []
  d = 2
  while d * d <= n:
    cnt = 0
    while n % d == 0:
      n //= d
      cnt += 1
    if cnt > 0:
      factors.append((d, cnt))
    d += 1 if d == 2 else 2
  if n > 1:
    factors.append((n, 1))
  return factors


def _next_pow2(N):
  """Smallest p such that 2**p >= 2*N + 1."""
  p = 1
  while (1 << p) < 2 * N + 1:
    p += 1
  return p


def _prime_sqrt_decomp(N):
  """
  Find i near floor(sqrt(N)) whose prime factors are a subset of N's.
  Used for balanced Cooley-Tukey splits.
  """
  fac = _get_prime_factors(N)
  if not fac:
    return 1
  i_sqrt = int(np.floor(np.sqrt(float(N))))
  primes = [f[0] for f in fac]
  powers = [f[1] for f in fac]

  for i in range(i_sqrt, 0, -1):
    rem = i
    ok = True
    for p, a_max in zip(primes, powers):
      cnt = 0
      while rem % p == 0:
        cnt += 1
        rem //= p
      if cnt > a_max:
        ok = False
        break
    if ok and rem > 1:
      if rem not in primes:
        ok = False
    if ok:
      return i
  return 1


def _is_pow2(n):
  """True if n is a power of 2."""
  if n <= 0:
    return False
  return (n & (n - 1)) == 0


def _is_prime(n):
  """True if n is prime."""
  if n < 2:
    return False
  if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
    return True
  if n % 2 == 0 or n % 3 == 0:
    return False
  d = 5
  while d * d <= n:
    if n % d == 0 or n % (d + 2) == 0:
      return False
    d += 6
  return True


def _balanced_split(N):
  """Split N = N1*N2 with N1 <= N2, balanced via prime_sqrt_decomp."""
  if N == 1:
    return 1, 1
  A = _prime_sqrt_decomp(N)
  B = N // A
  if A <= B:
    return A, B
  return B, A


###############################################################################
# Plan constants
###############################################################################

_NAIVE_TRUNC = 32

# Node type codes
_TP_IDENT = 0
_TP_N3 = 1
_TP_N4 = 2
_TP_NAIVE = 3
_TP_RADIX2 = 4
_TP_CT = 5
_TP_BLUESTEIN = 6


# Stack growth helpers (dynamic resizing via np.resize)

def _grow_bs(bs_n, bs_pid, bs_cp, min_size):
  """Double the capacity of BFS scratch arrays."""
  new_size = max(min_size, len(bs_n) * 2)
  return (np.resize(bs_n, new_size),
          np.resize(bs_pid, new_size),
          np.resize(bs_cp, new_size))


def _grow_exec_stack(st_id, st_bc, st_st, st_ph, min_size):
  """Double capacity of executor stack arrays."""
  new_size = max(min_size, len(st_id) * 2)
  return (np.resize(st_id, new_size),
          np.resize(st_bc, new_size),
          np.resize(st_st, new_size),
          np.resize(st_ph, new_size))


###############################################################################
# Twiddle factor construction
###############################################################################

def _twiddle_radix2(N):
  """
  Radix-2 twiddle: exp(-2*pi*i * k/N) for k = 0..N/2-1.
  Returns numpy object array of mpc values, shape (N//2,).
  """
  tw = np.empty(N // 2, dtype=object)
  for k in range(N // 2):
    tw[k] = mp.e ** (-2j * mp.pi * k / N)
  return tw


def _twiddle_ct(N1, N2):
  """
  Cooley-Tukey twiddle: exp(-2*pi*i * n1*k2 / (N1*N2)).
  Returns numpy object array of mpc values, shape (N1, N2).
  """
  tw = np.empty((N1, N2), dtype=object)
  for n1 in range(N1):
    for k2 in range(N2):
      tw[n1, k2] = mp.e ** (-2j * mp.pi * n1 * k2 / (N1 * N2))
  return tw


def _twiddle_bluestein(N, M):
  """
  Bluestein twiddle factors.
  Returns concatenated: tw_A (N,), tw_B (M,), radix2 twiddles (M//2,).

  tw_A[n] = exp(-pi*i * n*(n mod 2N) / N)
  tw_B: padded to M, values are conj(tw_A) mirrored
  """
  tw_A = np.empty(N, dtype=object)
  for n in range(N):
    n_mod = n % (2 * N)
    tw_A[n] = mp.e ** (-1j * mp.pi * n * n_mod / N)

  tw_B = np.empty(M, dtype=object)
  for i in range(M):
    d = i if i <= M - i else M - i
    if d < N:
      tw_B[i] = tw_A[d].conjugate()
    else:
      tw_B[i] = mp.mpc(0)

  rad2_tw = _twiddle_radix2(M)
  return np.concatenate((tw_A, tw_B, rad2_tw))


###############################################################################
# Plan construction
###############################################################################

def _build_plan(N):
  """
  Build a flat pre-order execution plan for an N-point FFT.

  Returns 8-tuple of numpy arrays:
    nodes_type : int8    (n_nodes,)  -- node type code
    nodes_N    : int64   (n_nodes,)  -- transform size
    nodes_p1   : int64   (n_nodes,)  -- extra param 1
    nodes_p2   : int64   (n_nodes,)  -- extra param 2
    nodes_c1   : int64   (n_nodes,)  -- left child  (-1 = leaf)
    nodes_c2   : int64   (n_nodes,)  -- right child (-1 = leaf)
    tw_data    : object  (tw_total,) -- concatenated twiddle factors
    tw_start   : int64   (n_nodes,)  -- offset into tw_data per node
  """
  # ---- Pass 1: count nodes and twiddle size ----
  _BS_INIT = 16
  bs_n = np.zeros(_BS_INIT, dtype=np.int64)
  bs_pid = -np.ones(_BS_INIT, dtype=np.int64)
  bs_cp = np.zeros(_BS_INIT, dtype=np.int8)

  n_nodes = 0
  tw_total = 0
  bsp = 0
  bs_n[0] = N
  bs_pid[0] = -2
  bsp = 1

  while bsp > 0:
    bsp -= 1
    n_val = int(bs_n[bsp])
    if n_val == 1:
      pass
    elif n_val == 3:
      pass
    elif n_val == 4:
      pass
    elif _is_pow2(n_val):
      tw_total += n_val // 2
    elif n_val < _NAIVE_TRUNC:
      pass
    elif _is_prime(n_val):
      M_val = 1 << _next_pow2(n_val)
      tw_total += n_val + M_val + M_val // 2
    else:
      N1, N2 = _balanced_split(n_val)
      tw_total += N1 * N2
      if bsp + 2 > len(bs_n):
        bs_n, bs_pid, bs_cp = _grow_bs(bs_n, bs_pid, bs_cp, bsp + 2)
      bs_n[bsp] = N2
      bs_pid[bsp] = -2
      bs_cp[bsp] = 0
      bsp += 1
      bs_n[bsp] = N1
      bs_pid[bsp] = -2
      bs_cp[bsp] = 0
      bsp += 1
    n_nodes += 1

  # Allocate plan arrays
  nodes_type = np.zeros(n_nodes, dtype=np.int8)
  nodes_N = np.zeros(n_nodes, dtype=np.int64)
  nodes_p1 = np.zeros(n_nodes, dtype=np.int64)
  nodes_p2 = np.zeros(n_nodes, dtype=np.int64)
  nodes_c1 = -np.ones(n_nodes, dtype=np.int64)
  nodes_c2 = -np.ones(n_nodes, dtype=np.int64)
  tw_data = (np.zeros(tw_total, dtype=object)
             if tw_total > 0 else np.zeros(1, dtype=object))
  tw_start = np.zeros(n_nodes, dtype=np.int64)
  tw_pos = 0
  node_counter = 0

  # ---- Pass 2: build plan ----
  bs_n = np.zeros(_BS_INIT, dtype=np.int64)
  bs_pid = -np.ones(_BS_INIT, dtype=np.int64)
  bs_cp = np.zeros(_BS_INIT, dtype=np.int8)
  bsp = 0
  bs_n[0] = N
  bs_pid[0] = -1
  bs_cp[0] = 0
  bsp = 1

  while bsp > 0:
    bsp -= 1
    n_val = int(bs_n[bsp])
    pid = int(bs_pid[bsp])
    cp = int(bs_cp[bsp])

    # Determine node type
    if n_val == 1:
      tp = _TP_IDENT
    elif n_val == 3:
      tp = _TP_N3
    elif n_val == 4:
      tp = _TP_N4
    elif _is_pow2(n_val):
      tp = _TP_RADIX2
    elif n_val < _NAIVE_TRUNC:
      tp = _TP_NAIVE
    elif _is_prime(n_val):
      tp = _TP_BLUESTEIN
    else:
      tp = _TP_CT

    nid = node_counter
    node_counter += 1
    nodes_type[nid] = tp
    nodes_N[nid] = n_val

    if tp == _TP_RADIX2:
      # Store log2(N) for use in bit-reversal
      p_val = 0
      tmp_val = n_val >> 1
      while tmp_val > 0:
        p_val += 1
        tmp_val >>= 1
      nodes_p1[nid] = p_val
      tw = _twiddle_radix2(n_val)
      tw_sz = tw.size
      tw_start[nid] = tw_pos
      tw_data[tw_pos:tw_pos + tw_sz] = tw
      tw_pos += tw_sz

    elif tp == _TP_CT:
      N1, N2 = _balanced_split(n_val)
      nodes_p1[nid] = N1
      nodes_p2[nid] = N2
      tw = _twiddle_ct(N1, N2).ravel()
      tw_sz = tw.size
      tw_start[nid] = tw_pos
      tw_data[tw_pos:tw_pos + tw_sz] = tw
      tw_pos += tw_sz
      # Push children: N2 first (right child), then N1 (left child)
      if bsp + 2 > len(bs_n):
        bs_n, bs_pid, bs_cp = _grow_bs(bs_n, bs_pid, bs_cp, bsp + 2)
      bs_n[bsp] = N2
      bs_pid[bsp] = nid
      bs_cp[bsp] = 1
      bsp += 1
      bs_n[bsp] = N1
      bs_pid[bsp] = nid
      bs_cp[bsp] = 0
      bsp += 1

    elif tp == _TP_BLUESTEIN:
      M_val = 1 << _next_pow2(n_val)
      # Compute log2(M) once at plan-build time
      log2_M = 0
      tmp_m = M_val >> 1
      while tmp_m > 0:
        log2_M += 1
        tmp_m >>= 1
      nodes_p1[nid] = M_val
      nodes_p2[nid] = log2_M
      tw = _twiddle_bluestein(n_val, M_val)
      tw_sz = tw.size
      tw_start[nid] = tw_pos
      tw_data[tw_pos:tw_pos + tw_sz] = tw
      tw_pos += tw_sz

    # Link child to parent
    if pid >= 0:
      if cp == 0:
        nodes_c1[pid] = nid
      else:
        nodes_c2[pid] = nid

  # Trim tw_data to actual size
  if tw_total > 0:
    tw_data = tw_data[:tw_total]

  return (nodes_type, nodes_N, nodes_p1, nodes_p2,
          nodes_c1, nodes_c2, tw_data, tw_start)


###############################################################################
# Leaf kernels
###############################################################################

def _fft_ident(x_2d, br, bc, st):
  """Identity: no-op for N=1."""
  pass


def _fft_3(x_2d, br, bc, st):
  """
  Unrolled DFT for N=3.
  w1 = exp(-2*pi*i/3) = -1/2 - i*sqrt(3)/2
  w2 = exp(-4*pi*i/3) = conj(w1)
  """
  sqrt3_half = mp.sqrt(3) / 2
  w1 = mp.mpc(-0.5, -sqrt3_half)
  w2 = mp.mpc(-0.5, sqrt3_half)

  a = x_2d[br, bc]
  b = x_2d[br, bc + st]
  c = x_2d[br, bc + 2 * st]

  x_2d[br, bc] = a + b + c
  x_2d[br, bc + st] = a + w1 * b + w2 * c
  x_2d[br, bc + 2 * st] = a + w2 * b + w1 * c


def _fft_4(x_2d, br, bc, st):
  """
  Unrolled DFT for N=4.
  Twiddle factors = {1, -i, -1, i} -- zero real multiplies.
  """
  a = x_2d[br, bc]
  b = x_2d[br, bc + st]
  c = x_2d[br, bc + 2 * st]
  d = x_2d[br, bc + 3 * st]

  ib = 1j * b
  id_ = 1j * d

  x_2d[br, bc] = a + b + c + d
  x_2d[br, bc + st] = a - ib - c + id_
  x_2d[br, bc + 2 * st] = a - b + c - d
  x_2d[br, bc + 3 * st] = a + ib - c - id_


def _fft_naive(x_2d, br, bc, st, N):
  """Direct O(N^2) Vandermonde DFT."""
  out = np.empty(N, dtype=object)
  for k in range(N):
    s = mp.mpc(0)
    for j in range(N):
      arg = mp.e ** (-2j * mp.pi * k * j / N)
      s += x_2d[br, bc + j * st] * arg
    out[k] = s
  for k in range(N):
    x_2d[br, bc + k * st] = out[k]


def _fft_radix2(x_2d, br, bc, st, N, tw, p):
  """
  Iterative Cooley-Tukey DIT radix-2 FFT for N = 2^p.
  tw: exp(-2*pi*i * k/N) for k = 0..N/2-1.
  p: log2(N), precomputed at plan-build time.
  """
  # Bit-reversal permutation
  for i in range(N):
    j = 0
    m = i
    for _ in range(p):
      j = (j << 1) | (m & 1)
      m >>= 1
    if j > i:
      tmp = x_2d[br, bc + i * st]
      x_2d[br, bc + i * st] = x_2d[br, bc + j * st]
      x_2d[br, bc + j * st] = tmp

  # Butterfly stages
  sz = 2
  while sz <= N:
    hsz = sz >> 1
    d_tw = N // sz
    for i in range(0, N, sz):
      for j_off in range(hsz):
        j = i + j_off
        k_tw = j_off * d_tw
        ev = x_2d[br, bc + j * st]
        od = x_2d[br, bc + (j + hsz) * st]
        t = tw[k_tw] * od
        x_2d[br, bc + j * st] = ev + t
        x_2d[br, bc + (j + hsz) * st] = ev - t
    sz <<= 1


def _fft_radix2_inplace(x_1d, N, tw, p):
  """
  Radix-2 DIT FFT on a contiguous 1D object array.
  Used internally by Bluestein. Operates in-place.
  p: log2(N), precomputed.
  """
  # Bit-reversal
  for i in range(N):
    j = 0
    m = i
    for _ in range(p):
      j = (j << 1) | (m & 1)
      m >>= 1
    if j > i:
      x_1d[i], x_1d[j] = x_1d[j], x_1d[i]

  # Butterfly
  sz = 2
  while sz <= N:
    hsz = sz >> 1
    d_tw = N // sz
    for i in range(0, N, sz):
      for j_off in range(hsz):
        j = i + j_off
        k_tw = j_off * d_tw
        ev = x_1d[j]
        od = x_1d[j + hsz]
        t = tw[k_tw] * od
        x_1d[j] = ev + t
        x_1d[j + hsz] = ev - t
    sz <<= 1


def _fft_bluestein(x_2d, br, bc, st, N, M, tw_all, log2_M):
  """
  Bluestein chirp-Z transform for prime N.
  Pads to M = 2^p >= 2N+1, performs convolution.
  tw_all contains: tw_A(N) + tw_B(M) + radix2_tw(M/2).
  log2_M is precomputed at plan-build time.
  """
  tw_A = tw_all[:N]
  tw_B = tw_all[N:N + M]
  radix2_tw = tw_all[N + M:]

  # Extract signal and multiply by tw_A
  sig = np.empty(N, dtype=object)
  for n in range(N):
    sig[n] = x_2d[br, bc + n * st] * tw_A[n]

  # Zero-pad to M
  y = np.zeros(M, dtype=object)
  for n in range(N):
    y[n] = sig[n]

  # Forward FFT of y
  _fft_radix2_inplace(y, M, radix2_tw, log2_M)

  # FFT of tw_B
  z_B = np.empty(M, dtype=object)
  for m_idx in range(M):
    z_B[m_idx] = tw_B[m_idx]
  _fft_radix2_inplace(z_B, M, radix2_tw, log2_M)

  # Convolution: Y .* FFT(tw_B)
  for m_idx in range(M):
    y[m_idx] *= z_B[m_idx]

  # Inverse FFT: conj(FFT(conj(y))) / M
  for m_idx in range(M):
    y[m_idx] = y[m_idx].conjugate()
  _fft_radix2_inplace(y, M, radix2_tw, log2_M)
  for m_idx in range(M):
    y[m_idx] = y[m_idx].conjugate() / M

  # Crop and multiply by tw_A
  for n in range(N):
    x_2d[br, bc + n * st] = y[n] * tw_A[n]


###############################################################################
# Cooley-Tukey helpers
###############################################################################

def _ct_twiddle_multiply(x_2d, br, bc, st, N1, N2, tw):
  """
  Multiply (N1,N2) block by twiddle factors in-place.
  tw is shape (N1, N2).
  """
  for k1 in range(N1):
    for k2 in range(N2):
      idx = bc + (k1 * N2 + k2) * st
      x_2d[br, idx] *= tw[k1, k2]


def _ct_transpose_output(x_2d, br, bc, st, N1, N2):
  """
  Transpose (N1,N2) block to (N2,N1) in-place.
  """
  tmp = np.empty((N1, N2), dtype=object)
  for k1 in range(N1):
    for k2 in range(N2):
      idx = bc + (k1 * N2 + k2) * st
      tmp[k1, k2] = x_2d[br, idx]

  for k2 in range(N2):
    for k1 in range(N1):
      idx = bc + (k2 * N1 + k1) * st
      x_2d[br, idx] = tmp[k1, k2]


###############################################################################
# Stack-based plan executor
###############################################################################

def _fft_exec(x_2d, plan):
  """
  Execute a FFT plan on x_2d shape (batch, N).
  Operates in-place.

  The plan is 8 arrays as returned by _build_plan().
  """
  (nodes_tp, nodes_n, nodes_p1, nodes_p2,
   nodes_c1, nodes_c2, tw_data, tw_start) = plan

  batch = x_2d.shape[0]

  # Stack per batch row
  _EXEC_STACK_INIT = 64
  st_id = np.zeros(_EXEC_STACK_INIT, dtype=np.int64)
  st_bc = np.zeros(_EXEC_STACK_INIT, dtype=np.int64)
  st_st = np.zeros(_EXEC_STACK_INIT, dtype=np.int64)
  st_ph = np.zeros(_EXEC_STACK_INIT, dtype=np.int8)

  for br in range(batch):
    sp = 0
    st_id[0] = 0  # root node
    st_bc[0] = 0
    st_st[0] = 1
    st_ph[0] = 0
    sp = 1

    while sp > 0:
      sp -= 1
      nid = int(st_id[sp])
      bc = int(st_bc[sp])
      st = int(st_st[sp])
      ph = int(st_ph[sp])

      tp = nodes_tp[nid]
      Ns = int(nodes_n[nid])

      if tp == _TP_IDENT:
        pass

      elif tp == _TP_N3:
        _fft_3(x_2d, br, bc, st)

      elif tp == _TP_N4:
        _fft_4(x_2d, br, bc, st)

      elif tp == _TP_NAIVE:
        _fft_naive(x_2d, br, bc, st, Ns)

      elif tp == _TP_RADIX2:
        p_val = int(nodes_p1[nid])
        tw = tw_data[tw_start[nid]:tw_start[nid] + Ns // 2]
        _fft_radix2(x_2d, br, bc, st, Ns, tw, p_val)

      elif tp == _TP_CT:
        N1 = int(nodes_p1[nid])
        N2 = int(nodes_p2[nid])
        c1 = int(nodes_c1[nid])
        c2 = int(nodes_c2[nid])

        if ph == 0:
          # Phase 0: push N2 children (column DFTs, size N1, stride N2)
          needed = sp + 1 + N2
          if needed > len(st_id):
            st_id, st_bc, st_st, st_ph = _grow_exec_stack(
                st_id, st_bc, st_st, st_ph, needed)
          st_id[sp] = nid
          st_bc[sp] = bc
          st_st[sp] = st
          st_ph[sp] = 1
          sp += 1
          for k in range(N2):
            st_id[sp] = c1
            st_bc[sp] = bc + k * st
            st_st[sp] = N2 * st
            st_ph[sp] = 0
            sp += 1

        elif ph == 1:
          # Phase 1: twiddle multiply
          tw = tw_data[tw_start[nid]:tw_start[nid] + N1 * N2].reshape(N1, N2)
          _ct_twiddle_multiply(x_2d, br, bc, st, N1, N2, tw)
          # Phase 2: push N1 children (row DFTs, size N2, stride 1)
          needed = sp + 1 + N1
          if needed > len(st_id):
            st_id, st_bc, st_st, st_ph = _grow_exec_stack(
                st_id, st_bc, st_st, st_ph, needed)
          st_id[sp] = nid
          st_bc[sp] = bc
          st_st[sp] = st
          st_ph[sp] = 2
          sp += 1
          for k in range(N1):
            st_id[sp] = c2
            st_bc[sp] = bc + k * N2 * st
            st_st[sp] = st
            st_ph[sp] = 0
            sp += 1

        elif ph == 2:
          # Phase 2: transpose (N1,N2) -> (N2,N1)
          _ct_transpose_output(x_2d, br, bc, st, N1, N2)

      elif tp == _TP_BLUESTEIN:
        M_val = int(nodes_p1[nid])
        log2_M_val = int(nodes_p2[nid])
        tw = tw_data[tw_start[nid]:
                     tw_start[nid] + Ns + M_val + M_val // 2]
        _fft_bluestein(x_2d, br, bc, st, Ns, M_val, tw, log2_M_val)


#
# :D
#
