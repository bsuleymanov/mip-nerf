import torch


def safe_trig_helper(x, fn, t=100 * torch.pi):
  return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.cos)