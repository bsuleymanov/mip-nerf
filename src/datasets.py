import torch
from pathlib import Path
from PIL import Image
import numpy as np
import collections
from . import utils
import torch.utils.data.distributed as data_dist
from hydra.utils import to_absolute_path, get_original_cwd


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions


class Dataset:
    def __init__(self, data_dir, split, near, far,
                 to_render_path, batching_mode, factor,
                 to_spherify, llffhold):
        self.dataset = []
        self.data_dir = Path(data_dir)
        self.split = split
        self.near = near
        self.far = far
        self.batching_mode = batching_mode
        self.factor = factor
        self.to_render_path = to_render_path
        self.to_spherify = to_spherify
        self.llffhold = llffhold

        if split == "train":
            self._train_init()
        elif split == "test":
            self._test_init()
        else:
            raise ValueError(
                f"the split argument should be either 'train' or 'test', set"
                f"to {split} here.")

    def _train_init(self):
        self._load_renderings()
        self._generate_rays()
        if self.batching_mode == "all_images":
            self.images = self.images.reshape([-1, 3])
            self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                             self.rays)
        # TODO: Implement single_image batching_mode.
        #elif #self.batching_mode == "single_image":
            #self.images = self.images.reshape([-1, self.resolution, 3])
            #self.rays = utils.namedtuple_map(
            #    lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)

        else:
            raise NotImplementedError(
                f"{self.batching_mode} batching strategy is not implemented.")

    def _test_init(self):
        self._load_renderings()
        self._generate_rays()
        self.it = 0

    def _next_train(self):
        # implement it in Dataloader class
        ...

    def _next_test(self):
        # implement it in Dataloader class
        ...


    def _generate_rays(self):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)
        directions = ((camera_dirs[None, ..., None, :] *
                       self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                                  directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(
            np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])
        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)

    def __getitem__(self, index):
        if self.split == "train":
            if self.batching_mode == "all_images":
                pixels = self.images[index]
                rays = utils.namedtuple_map(lambda r: r[index], self.rays)
            # TODO: Implement single_image batching_mode.
            else:
                raise NotImplementedError(
                    f"{self.batching} batching strategy is not implemented.")
            return {"pixels": pixels, "rays": rays}
        elif self.split == "test":
            if self.to_render_path:
                return {"rays": utils.namedtuple_map(lambda r: r[index], self.render_rays)}
            else:
                return {
                    'pixels': self.images[index],
                    'rays': utils.namedtuple_map(lambda r: r[index], self.rays)
                }


    def __len__(self):
        return self.n_examples


class TrainDataset:
    ...

class TestDataset:
    ...


class LLFFDataset(Dataset):
    # def __init__(self):
    #     super(LLFFDataset, self).__init__()

    def _load_renderings(self):
        #self.data_dir = Path(data_dir)
        if self.factor > 1:
            image_dir = self.data_dir / f"images_{self.factor}"
        else:
            image_dir = self.data_dir / f"images"
        if not image_dir.exists():
            raise ValueError(f'Image folder {str(image_dir)} does not exist.')
        extensions = ("jpg", "JPG", "png")
        image_paths = []
        for ext in extensions:
            image_paths.extend(sorted(list(map(str, image_dir.glob(f"*.{ext}")))))
        images = []
        for image_path in image_paths:
            print(image_path)
            with open(image_path, "rb") as image_obj:
                image = np.array(Image.open(image_obj), dtype=np.float32) / 255.
                images.append(image)
        images = np.stack(images, axis=-1)

        # load poses and bds
        with open(self.data_dir / "poses_bounds.npy", "rb") as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError(f"Mismatch between images {image.shape[-1]} "
                               f"and poses {poses.shape[-1]}.")

        poses[:2, 4, :] = np.array(image.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale

        poses = self._recenter_poses(poses)

        if self.to_spherify:
            poses = self._generate_spherical_poses(poses, bds)

        if not self.to_spherify and self.split == "test":
            self._generate_spiral_poses(poses, bds)

        # select the split
        i_test = np.arange(images.shape[0])[::self.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test])
        if self.split == 'train':
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]

        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        if self.to_render_path:
            self.n_examples = self.render_poses.shape[0]
        else:
            self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        if self.split == 'test':
            n_render_poses = self.render_poses.shape[0]
            self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                              axis=0)

        super()._generate_rays()

        if not self.to_spherify:
            ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                         self.rays.directions,
                                                         self.focal, self.w, self.h)

            mat = ndc_origins
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
            dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

            dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
            dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

            ones = np.ones_like(ndc_origins[..., :1])
            self.rays = Rays(
                origins=ndc_origins,
                directions=ndc_directions,
                viewdirs=self.rays.directions,
                radii=radii,
                lossmult=ones,
                near=ones * self.near,
                far=ones * self.far)

        # Split poses from the dataset and generated poses
        if self.split == 'test':
            self.camtoworlds = self.camtoworlds[n_render_poses:]
            split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            split0, split1 = zip(*split)
            self.render_rays = Rays(*split0)
            self.rays = Rays(*split1)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable 'focus depth' for this dataset.
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(c2w[:3, :4], (np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate([
            new_poses,
            np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        if self.split == 'test':
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset



class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class LLFFDataloader:
    def __init__(self, data_dir, split, near, far,
                 to_render_path, batching_mode, factor,
                 to_spherify, llffhold, batch_size=8,
                 drop_last=False, shuffle=False,
                 num_workers=8, world_size=1, rank=0):
        data_dir = Path(to_absolute_path(data_dir))
        self.dataset = LLFFDataset(
            data_dir, split, near, far,
            to_render_path, batching_mode, factor,
            to_spherify, llffhold)
        self.data_sampler = data_dist.DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        self.loader = FastDataLoader(
            dataset=self.dataset, batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            sampler=self.data_sampler)