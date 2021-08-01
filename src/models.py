import torch
from torch import nn
import mip
from torch.nn import functional as F
import utils


class MipNerfModel(nn.Module):
    def __init__(self,
                 n_samples: int = 128,
                 n_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_level_grad: bool = True,
                 use_viewdirs: bool = True,
                 lindisp: bool = False,
                 ray_shape: str = "cylinder",
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_noise: float = 1.,
                 density_bias: float = -1.,
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False):
        super(MipNerfModel, self).__init__()
        self.mlp = MLP()
        self.n_levels = n_levels
        self.stop_level_grad = stop_level_grad
        self.deg_view = deg_view
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
        device = rays.origins.device
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
                self.max_deg_point,
                device=device
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
                raw_density += self.density_noise * torch.randn(
                    *(raw_density.shape), dtype=raw_density.dtype, device=raw_density.device)

            rgb = F.sigmoid(raw_rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = F.softplus(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights = mip.volumetric_rendering(
                rgb, density, t_vals, rays.directions, white_bkgd=white_bg)
            ret.append((comp_rgb, distance, acc))

        return ret


def make_mipnerf(example_rays, device, randomized, white_bg):
    model = MipNerfModel().to(device)
    model(example_rays, randomized, white_bg)

    return model



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
        self.n_layers = n_layers
        self.skip_layer = skip_layer
        self.n_density_channels = n_density_channels
        self.n_rgb_channels = n_rgb_channels
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(DenseBlock(n_units))
        self.density_layer = DenseBlock(n_density_channels)
        #if condition is not None:
        self.bottleneck_layer = DenseBlock(n_units)
        self.cond_layers = nn.ModuleList()
        for i in range(n_layers_condition):
            self.cond_layers.append(DenseBlock(n_units_condition))
        self.cond_layers = nn.Sequential(*self.cond_layers)
        self.rgb_layer = DenseBlock(n_rgb_channels)

    def forward(self, x, condition):
        feature_dim = x.size(-1)
        n_samples = x.size(1)
        x = x.reshape(-1, feature_dim)
        inputs = x
        for i in range(self.n_layers):
            x = self.layers[i](x)
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
            x = self.cond_layers(x)

        raw_rgb = self.rgb_layer(x).reshape(
            -1, n_samples, self.n_rgb_channels)

        return raw_rgb, raw_density


def render_image(render_fn, rays, rank, chunk=8192):
    n_devices = torch.cuda.device_count()
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = utils.namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)

    #host_id = jax.host_id()
    #n_hosts = 1
    #host_id = 0
    results = []
    for i in range(0, num_rays, chunk):
        # pylint: disable=cell-var-from-loop
        chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
        chunk_size = chunk_rays[0].shape[0]
        rays_remaining = chunk_size % torch.cuda.device_count()
        if rays_remaining != 0:
            padding = n_devices - rays_remaining
            chunk_rays = utils.namedtuple_map(
                # mode = "edge", not reflect
                lambda r: F.pad(r, (0, padding, 0, 0), mode='reflect'), chunk_rays)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by
        # host_count.
        rays_per_host = chunk_rays[0].shape[0]
        start, stop = 0 * rays_per_host, (0 + 1) * rays_per_host
        chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]),
                                          chunk_rays)
        chunk_results = render_fn(chunk_rays)[-1]
        results.append([utils.unshard(x[0], padding) for x in chunk_results])
        # pylint: enable=cell-var-from-loop
    rgb, distance, acc = [torch.cat(r, axis=0) for r in zip(*results)]
    rgb = rgb.reshape((height, width, -1))
    distance = distance.reshape((height, width))
    acc = acc.reshape((height, width))
    return (rgb, distance, acc)