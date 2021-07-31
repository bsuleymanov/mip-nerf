import torch
from torch import nn
import mip
from torch.nn import functional as F


class MipNerfModel(nn.Module):
    def __init__(self,
                 n_samples: int = 128,
                 n_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_level_grad: bool = True,
                 use_viewdirs: bool = True,
                 lindisp: bool = False,
                 ray_shape: str = "cone",
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False):
        super(MipNerfModel, self).__init__()
        self.mlp = MLP()
        self.n_samples = n_samples
        self.lindisp = lindisp
        self.ray_shape = ray_shape
        self.resample_padding = resample_padding
        self.disable_integration = disable_integration
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.use_viewdirs = use_viewdirs
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.density_bias = density_bias

    def forward(self, rays, randomized, white_bg):
        ret = []
        for i_level in range(self.n_levels):
            if i_level == 0:
                # stratified sampling along rays
                t_vals, samples = mip.sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.n_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.lindisp,
                    self.ray_shape
                )
            else:
                t_vals, samples = mip.resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_vals,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_level_grad,
                    resample_padding=self.resample_padding,
                )
            if self.disable_integration:
                samples = (samples[0], torch.zeros_like(samples[1]))
            samples_enc = mip.integrated_pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point
            )

            if self.use_viewdirs:
                viewdirs_enc = mip.pos_enc(
                    rays.viewdirs,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                )
                raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_density = self.mlp(samples_enc)

            if randomized and self.density_noise > 0:
                raw_density += self.density_noise * torch.normal(
                    raw_density.shape, dtype=raw_density.dtype)

            rgb = F.sigmoid(raw_rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = F.softplus(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights = mip.volumetric_rendering(
                rgb, density, t_vals, rays.directions, white_bg=white_bg)
            ret.append((comp_rgb, distance, acc))

        return ret


class DenseBlock(nn.Module):
    def __init__(self, n_units: int = 256):
        super(DenseBlock, self).__init__()
        layers = [
            nn.LazyLinear(n_units), # use glorot uniform init
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self,
                 n_layers: int = 8,
                 n_units: int = 256,
                 n_layers_condition: int = 1,
                 n_units_condition: int = 128,
                 skip_layer: int = 4,
                 n_rgb_channels: int = 3,
                 n_density_channels: int = 1,
                 condition = None):
        super(MLP, self).__init__()
        self.n_density_channels = n_density_channels
        self.n_rgb_channels = n_rgb_channels
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(DenseBlock(n_units))
        self.density_layer = DenseBlock(n_density_channels)
        if self.condition is not None:
            self.bottleneck_layer = DenseBlock(n_units)
            self.cond_layers = nn.ModuleList()
            for i in range(n_layers_condition):
                self.cond_layers.append(DenseBlock(n_units_condition))
            self.cond_layers = nn.Sequential(*self.cond_layers)
        self.rgb_layer = DenseBlock(n_rgb_channels)

    def forward(self, x, condition=None):
        feature_dim = x.size(-1)
        n_samples = x.size(1)
        x = x.reshape(-1, feature_dim)
        inputs = x
        for i in range(self.n_layers):
            x = self.layers[i]
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x).reshape(
            -1, n_samples, self.n_density_channels)

        if condition is not None:
            bottleneck = self.bottleneck_layer(x)
            condition = torch.tile(condition[:, None, :],
                                   (1, n_samples, 1))
            condition = condition.reshape([-1, condition.size(-1)])
            x = torch.cat([bottleneck, condition], dim=-1)
            x = self.cond_layers()

        raw_rgb = self.rgb_layer(x).reshape(
            -1, n_samples, self.n_rgb_channels)

        return raw_rgb, raw_density


def render_image(render_fn, rays, rng, chunk=8192):
    ...