import torch
from src import math


def pos_enc(x, min_deg, max_deg, append_identity=True):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)])
    xb = torch.reshape((x[..., None, :] * scales[:, None]),
                       list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * math.safe_sin(x)
    y_var = torch.maximum(
        0, 0.5 * (1 - torch.exp(-2 * x_var) * math.safe_cos(2 * x)) - y**2)
    return y, y_var


def lift_gaussian():
    ...


def conical_frustum_to_gaussian():
    ...


def cylinder_to_gaussian():
    ...


def cast_rays():
    ...


def integrated_pos_enc():
    ...


def volumetric_rendering():
    ...


def sample_along_rays():
    ...


def resample_along_rays():
    ...