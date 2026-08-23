import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import open3d as o3d
import numpy as np
import torch.nn.functional as F

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    'm1_15000':'m1','it':'it','ct':'ct','pt':'pt',
    # MICrONS cortical dendrites (see tools/swc_to_morphogen_npy.py).
    # The loader globs <root>/<synset>/<split>/*.npy and derives the conditioning
    # label from the synset INDEX, so the unconditional arm uses one synset and the
    # class-conditional arm uses one synset per cell class.
    'neurons':'neurons',
    'class_0':'class_0', 'class_1':'class_1', 'class_2':'class_2', 'class_3':'class_3',
    'class_4':'class_4', 'class_5':'class_5', 'class_6':'class_6',
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}



def _dataset_mean_std(arr, per_axis, block=256):
    """Dataset-wide mean and std of (N, P, C) without a full-size temporary.

    Matches arr.reshape(-1, C).mean(axis=0) and, for the std, either
    arr.reshape(-1, C).std(axis=0) (per_axis) or arr.reshape(-1).std(axis=0),
    i.e. the scalar std about the scalar mean.
    """
    n, p, c = arr.shape
    s1 = np.zeros(c, dtype=np.float64)
    s2 = np.zeros(c, dtype=np.float64)
    for i in range(0, n, block):
        b = arr[i:i + block].reshape(-1, c)
        # dtype= accumulates in float64 without allocating a float64 copy of the
        # block, and einsum avoids a full-size square temporary for the sum of
        # squares. Together these keep the extra memory at zero.
        s1 += b.sum(axis=0, dtype=np.float64)
        s2 += np.einsum('ij,ij->j', b, b, dtype=np.float64)
    m = n * p                      # samples per axis
    mean = s1 / m
    if per_axis:
        std = np.sqrt(np.maximum(s2 / m - mean * mean, 0.0))
    else:
        gmean = s1.sum() / (m * c)
        std = np.sqrt(np.maximum(s2.sum() / (m * c) - gmean * gmean, 0.0))
        std = np.asarray(std).reshape(1)
    return mean, std


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, box_per_shape=False,
                 random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3, use_mask=False):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.use_mask = use_mask
        self.box_per_shape = box_per_shape
        if use_mask:
            self.mask_transform = PointCloudMasks(radius=5, elev=5, azim=90)

        # Enumerate first, load second. Upstream accumulated every cloud in a
        # Python list, shuffled the list, then np.concatenate'd it -- which holds
        # the list and the full result alive simultaneously. On a 22,773-neuron
        # corpus the result alone is 4.1 GB, so the assembly peaked near 8.2 GB,
        # and DDPM_train loaded the corpus twice on top of that. Loading straight
        # into the destination row removes the duplicate entirely.
        entries = []  # (cate_idx, subd, mid)
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>"
                entries.append((cate_idx, subd, os.path.join(self.split, x[:-len('.npy')])))

        if not entries:
            raise RuntimeError(
                f'no .npy files found under {root_dir}/<{",".join(self.subdirs)}>/{self.split}')

        # Shuffle the index deterministically (based on the number of examples).
        # Applied while loading rather than afterwards; identical result whenever
        # every file is readable, which the warning below now makes visible.
        self.shuffle_idx = list(range(len(entries)))
        random.Random(38383).shuffle(self.shuffle_idx)

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = None
        n_written, n_unreadable = 0, 0
        for i in self.shuffle_idx:
            cate_idx, subd, mid = entries[i]
            obj_fname = os.path.join(root_dir, subd, mid + ".npy")
            try:
                point_cloud = np.load(obj_fname)  # (15k, 3)
            except Exception as exc:
                # Upstream swallowed this with a bare `except: continue`, so a
                # truncated or unreadable file silently shrank the training set
                # with nothing in the logs to show it.
                n_unreadable += 1
                if n_unreadable <= 5:
                    print(f'  WARNING: unreadable, skipping: {obj_fname} '
                          f'({type(exc).__name__}: {exc})')
                continue

            assert point_cloud.shape[0] == 15000
            if self.all_points is None:
                self.all_points = np.empty((len(entries),) + point_cloud.shape,
                                           dtype=point_cloud.dtype)
            self.all_points[n_written] = point_cloud
            n_written += 1
            self.cate_idx_lst.append(cate_idx)
            self.all_cate_mids.append((subd, mid))

        if n_unreadable:
            print(f'  WARNING: {n_unreadable}/{len(entries)} .npy files under '
                  f'{self.split} were unreadable and were dropped')
        if n_written == 0:
            raise RuntimeError(f'every .npy under {root_dir} for split {self.split} was unreadable')
        self.all_points = self.all_points[:n_written]
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        elif self.box_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.min(axis=1).reshape(B, 1, input_dim)

            self.all_points_std = self.all_points.max(axis=1).reshape(B, 1, input_dim) - self.all_points.min(axis=1).reshape(B, 1, input_dim)

        else:  # normalize across the dataset
            # Streamed in blocks. np.std materialises (x - mean)**2 at full size,
            # which is another 4.1 GB temporary on a 22,773-neuron corpus -- the
            # last full-size copy in the load path. Accumulating sums in float64
            # over blocks costs one block instead, and is if anything the more
            # accurate of the two (agreement with np.std measured at <1e-6
            # relative on real data).
            _mean, _std = _dataset_mean_std(self.all_points, normalize_std_per_axis)
            self.all_points_mean = _mean.astype(self.all_points.dtype).reshape(1, 1, input_dim)
            self.all_points_std = _std.astype(self.all_points.dtype).reshape(
                1, 1, input_dim if normalize_std_per_axis else 1)

        # In place: the out-of-place form allocated a third full-size temporary.
        self.all_points -= self.all_points_mean
        self.all_points /= self.all_points_std
        if self.box_per_shape:
            self.all_points -= 0.5
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape or self.box_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s


        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        out = {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': m, 'std': s, 'cate_idx': cate_idx,
            'sid': sid, 'mid': mid
        }

        if self.use_mask:
            # masked = torch.from_numpy(self.mask_transform(self.all_points[idx]))
            # ss = min(masked.shape[0], self.in_tr_sample_size//2)
            # masked = masked[:ss]
            #
            # tr_mask = torch.ones_like(masked)
            # masked = torch.cat([masked, torch.zeros(self.in_tr_sample_size - ss, 3)],dim=0)#F.pad(masked, (self.in_tr_sample_size-masked.shape[0], 0), "constant", 0)
            #
            # tr_mask =  torch.cat([tr_mask, torch.zeros(self.in_tr_sample_size- ss, 3)],dim=0)#F.pad(tr_mask, (self.in_tr_sample_size-tr_mask.shape[0], 0), "constant", 0)
            # out['train_points_masked'] = masked
            # out['train_masks'] = tr_mask
            tr_mask = self.mask_transform(tr_out)
            out['train_masks'] = tr_mask

        return out


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, box_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None,
                 use_mask=False):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape, box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3, use_mask=use_mask)



class PointCloudMasks(object):
    '''
    render a view then save mask
    '''
    def __init__(self, radius : float=10, elev: float =45, azim:float=315, ):

        self.radius = radius
        self.elev = elev
        self.azim = azim


    def __call__(self, points):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera = [self.radius * np.sin(90-self.elev) * np.cos(self.azim),
                  self.radius * np.cos(90 - self.elev),
                  self.radius * np.sin(90 - self.elev) * np.sin(self.azim),
                  ]
        # camera = [0,self.radius,0]
        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        mask = torch.zeros_like(points)
        mask[pt_map] = 1

        return mask #points[pt_map]


####################################################################################


