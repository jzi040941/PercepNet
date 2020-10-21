def vorbis_window(window_length, dtype=dtypes.float32, name=None):
  """Generate a [Vorbis power complementary window][vorbis].
  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.
  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.
  [vorbis]:
    https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform#Window_functions
  """
  window_length = _check_params(window_length, dtype)
  arg = math_ops.cast(math_ops.range(window_length), dtype=dtype)
  window = math_ops.sin(np.pi / 2.0 * math_ops.pow(math_ops.sin(
      np.pi / math_ops.cast(window_length, dtype=dtype) *
      (arg + 0.5)), 2.0))
  return window


def _len_guards(M):
  """Handle small or incorrect window lengths"""
  if int(M) != M or M < 0:
      raise ValueError('Window length M must be a non-negative integer')
  return M <= 1


def _extend(M, sym):
  """Extend window by 1 sample if needed for DFT-even symmetry"""
  if not sym:
      return M + 1, True
  else:
      return M, False


def _truncate(w, needed):
  """Truncate window by 1 sample if needed for DFT-even symmetry"""
  if needed:
      return w[:-1]
  else:
      return w

def vorbis(M,sym=True):

  if _len_guards(M):
      return np.ones(M)
  M, needs_trunc = _extend(M, sym)

  fac = (np.arange(0,M+1) + 0.5)/M*np.pi
  w = np.zeros(M)
  w += np.pi / 2 * np.power(np.sin(fac),2)

  return _truncate(w, needs_trunc)
