import torch
from src import math_ops
import math
import random


def pos_enc(x, min_deg, max_deg, append_identity=True):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=x.device)
    xb = torch.reshape((x[..., None, :] * scales[:, None]),
                       list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * math_ops.safe_sin(x)
    y_var = torch.maximum(
        torch.zeros_like(x_var, device=x_var.device), 0.5 * (1 - torch.exp(-2 * x_var) * math_ops.safe_cos(2 * x)) - y ** 2)
    return y, y_var




def lift_gaussian(d, t_mean, t_var, r_var, diag):
  mean = d[..., None, :] * t_mean[..., None]

  small_tensor = torch.ones_like(torch.sum(d**2, dim=-1, keepdims=True)) * 1e-10
  d_mag_sq = torch.maximum(small_tensor, torch.sum(d**2, dim=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = torch.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  if stable:
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                      (3 * mu**2 + hw**2)**2)
    r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                              (hw**4) / (3 * mu**2 + hw**2))
  else:
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
  t0 = t_vals[..., :-1]
  t1 = t_vals[..., 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    assert False
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[..., None, :]
  return means, covs


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True, device='0'):
  if diag:
    x, x_cov_diag = x_coord
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
  else:
    x, x_cov = x_coord
    num_dims = x.shape[-1]
    basis = torch.cat(
        [2**i * torch.eye(num_dims) for i in range(min_deg, max_deg)], 1)
    y = torch.matmul(x, basis)
    # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
    # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
    y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)

  return expected_sin(
      torch.cat([y, y + 0.5 * math.pi], dim=-1),
      torch.cat([y_var] * 2, dim=-1))[0]


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
  t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
  t_dists = t_vals[..., 1:] - t_vals[..., :-1]
  delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
  # Note that we're quietly turning density from [..., 0] to [...].
  density_delta = density[..., 0] * delta

  alpha = 1 - torch.exp(-density_delta)
  trans = torch.exp(-torch.cat([
      torch.zeros_like(density_delta[..., :1]),
      torch.cumsum(density_delta[..., :-1], dim=-1)
  ],
                                   dim=-1))
  weights = alpha * trans

  comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
  acc = weights.sum(dim=-1)
  distance = (weights * t_mids).sum(dim=-1) / acc
  distance = torch.clip(
      torch.nan_to_num(distance, float('inf')), t_vals[:, 0], t_vals[:, -1])
  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[..., None])
  return comp_rgb, distance, acc, weights


def sample_along_rays(origins, directions, radii, num_samples, near, far,
                      randomized, lindisp, ray_shape):
  batch_size = origins.shape[0]
  device = origins.device

  t_vals = torch.linspace(0., 1., num_samples + 1, device=device)
  if lindisp:
    t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  else:
    t_vals = near * (1. - t_vals) + far * t_vals

  if randomized:
    mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    upper = torch.cat([mids, t_vals[..., -1:]], -1)
    lower = torch.cat([t_vals[..., :1], mids], -1)
    t_rand = torch.rand(batch_size, num_samples + 1, device=device)
    t_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast t_vals to make the returned shape consistent.
    t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])
  means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
  return t_vals, (means, covs)


def resample_along_rays(origins, directions, radii, t_vals, weights,
                        randomized, ray_shape, stop_grad, resample_padding):
  weights_pad = torch.cat([
      weights[..., :1],
      weights,
      weights[..., -1:],
  ],
                                dim=-1)
  weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
  weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

  # Add in a constant (the sampling function will renormalize the PDF).
  weights = weights_blur + resample_padding

  new_t_vals = math_ops.sorted_piecewise_constant_pdf(
      t_vals,
      weights,
      t_vals.shape[-1],
      randomized,
  )
  if stop_grad:
    new_t_vals = new_t_vals.detach()
  means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
  return new_t_vals, (means, covs)