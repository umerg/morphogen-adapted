import os
import torch
import pandas as pd
from pprint import pprint
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from models.dit3d import DiT3D_models
from utils.misc import Evaluator
import numpy as np
import warnings
import json
from scipy.spatial import distance_matrix, KDTree
from utils.ske_connect import *
import heapq
import random
from sklearn.decomposition import PCA
from utils.cut import filter_short_branches
from utils.utils import load_neuron
from utils.swc_denoise import auxi
'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, y, clip_denoised: bool, return_pred_xstart: bool):
        
        model_output = denoise_fn(data, t, y)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        eps = eps.to(x_t.device)
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, y, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, y=y, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, y,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def reconstruct(self, x0, t, y, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)
        
        # DiT-3d
        self.model = DiT3D_models[args.model_type](input_size=args.voxel_size, num_classes=args.num_classes)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, y, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, y)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None, y=None):
        B, D, N = data.shape                           # [16, 3, 2048]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor
    return output
#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        # BUGFIX: upstream passed `train_dataset` here, so generation ran over the
        # TRAIN split rather than the held-out split.
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler

def neuron_swc_generator(skeleton, soma_radius=5.0, detect_radius=None, root_cap=2,
                         n_root_children=None, stem_sep=0.7,
                         gamma_seed=1.0, gamma_main=1.2, seed_direction=None):
    """Reconstruct a tree from skeleton points.

    `soma_radius` is written into the SWC radius column for the soma node and is
    a physical value in microns (5.0 um is anatomically sensible). Upstream also
    reused it as the density-detection radius -- but the skeleton reaching this
    function has been through pc_normlize(), so it lives on the unit sphere where
    the maximum pairwise distance is <= 2. A 5.0 query radius therefore returned
    EVERY point for every candidate, all densities tied, and np.argmax returned
    index 0, i.e. whatever point fps() happened to seed with. The root was
    effectively random.

    `detect_radius` separates the two roles. Express 5 um in normalized space as
    5.0 / m, where m is the neuron's max radial norm before normalization; for
    the MICrONS cortical corpus median m ~ 244 um, giving ~0.02, which is also
    where soma-placement error is empirically minimised.

    `root_cap` restores the behaviour described in the paper (Sec. 4.2): "each
    bifurcation, EXCLUDING THE SOMA, is limited to a maximum of two child
    branches". Upstream applied the cap to node 0 as well.
    """

    def detect_soma(points, radius):
        tree = KDTree(points)
        densities = np.array([len(tree.query_ball_point(p, radius)) for p in points])
        if densities.max() == densities.min():
            warnings.warn(
                'detect_soma: all local densities are identical at radius={:g} on data '
                'of extent {:g} -- the soma is being chosen arbitrarily. Pass a '
                'detect_radius appropriate to the coordinate scale.'.format(
                    radius, float(np.ptp(points, axis=0).max())),
                RuntimeWarning, stacklevel=2)
        return points[np.argmax(densities)]

    soma_center = detect_soma(skeleton, soma_radius if detect_radius is None else detect_radius)


    class NeuronTree:
        def __init__(self, root):
            self.nodes = {0: {'id': 0, 'pos': root, 'children': [], 'parent': None}}
            self.current_id = 1
            self.direction = np.zeros(3)  

        def add_node(self, parent_id, position):
            self.nodes[self.current_id] = {
                'id': self.current_id,
                'pos': position,
                'children': [],
                'parent': parent_id
            }
            self.nodes[parent_id]['children'].append(self.current_id)
            self.current_id += 1

        def update_direction(self):
       
            if len(self.nodes) > 1:
                positions = np.array([n['pos'] for n in self.nodes.values()])
                pca = PCA(n_components=1)
                pca.fit(positions)
                self.direction = pca.components_[0]
            else:
                self.direction = np.random.randn(3) 
                self.direction /= np.linalg.norm(self.direction) + 1e-6

  
    tree = NeuronTree(soma_center)
    candidate_edges = []
    dists = distance_matrix([soma_center], skeleton)[0]

    visited = set([0])
    skeleton_flags = np.zeros(len(skeleton), dtype=bool)
    skeleton_flags[np.argmin(dists)] = True

    if n_root_children and n_root_children > 1:
        # --- explicit multi-stem seeding -------------------------------------
        # The greedy loop below cannot grow a realistic soma on its own. Soma edges
        # are priced ONCE, at init, while `tree.direction` is still zeros, so they
        # never receive the directional discount that every frontier edge gets:
        #     soma edge     = d * (1.0 - 0)        = 1.0*d      (fixed forever)
        #     frontier edge = d * (1.2 - cos_sim)  = 0.2..2.2*d (re-priced each step)
        # An aligned frontier edge is therefore ~5x cheaper than any soma edge at
        # the same distance, so after its first child the soma essentially never
        # wins another contest -- root degree collapses to 1-3 regardless of the cap.
        # Seeding the k stems directly sidesteps the pricing asymmetry entirely.
        #
        # Stems are chosen by DIRECTION, not proximity: real primary dendrites leave
        # the soma on distinct bearings, whereas the k nearest points typically all
        # sit on one dendrite.
        shell = np.argsort(dists)[1:max(n_root_children * 12, 60)]
        vecs = skeleton[shell] - soma_center
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / np.maximum(norms, 1e-9)

        order = sorted(zip(shell, unit), key=lambda t: dists[t[0]])
        picked = []
        # Progressively relax the bearing-separation requirement until k stems are
        # found. A fixed threshold silently under-delivers on neurons whose stems
        # leave at shallow angles, which would reintroduce the very bias we are
        # trying to remove.
        sep = stem_sep
        while len(picked) < n_root_children and sep < 0.99:
            for idx, u in order:
                if len(picked) >= n_root_children:
                    break
                if idx in {j for j, _ in picked}:
                    continue
                if all(float(np.dot(u, pu)) < sep for _, pu in picked):
                    picked.append((idx, u))
            sep += 0.1

        for idx, _ in picked:
            if skeleton_flags[idx]:
                continue
            tree.add_node(0, skeleton[idx])
            nid = tree.current_id - 1
            visited.add(nid)
            skeleton_flags[idx] = True
            npos = skeleton[idx]
            nd = distance_matrix([npos], skeleton)[0]
            for i, d in enumerate(nd):
                if not skeleton_flags[i] and d > 0:
                    heapq.heappush(candidate_edges, (d * 1.2, nid, i))
        # the soma is now fully populated; the loop must not add more stems
        root_cap = len(tree.nodes[0]['children'])

    # Soma edges are priced ONCE, here, and never re-priced. Upstream did so with
    # `tree.direction` still zeros, so cos_sim == 0 and every soma edge is charged the
    # full gamma_seed*d -- while frontier edges, re-pushed with the live direction, can
    # be discounted to (gamma_main-1)*d. `seed_direction='cloud'` estimates d_g from the
    # skeleton's principal axis instead, which is what the paper's
    # "current principal morphological direction" reduces to when the tree is one node.
    seed_dir = tree.direction
    if seed_direction == 'cloud' and len(skeleton) > 2:
        seed_dir = PCA(n_components=1).fit(skeleton).components_[0]

    for i, d in enumerate(dists):
        if d > 0 and not skeleton_flags[i]:
            edge = skeleton[i] - soma_center
            cos_sim = np.dot(edge, seed_dir) / (np.linalg.norm(edge) + 1e-6)
            # |cos_sim| so a stem is cheap whether it leaves along +d_g or -d_g;
            # a signed term would price the basal stems out of contention entirely.
            if seed_direction == 'cloud':
                cos_sim = abs(cos_sim)
            heapq.heappush(candidate_edges, (d * (gamma_seed - cos_sim), 0, i))

    while candidate_edges:
        weight, parent_id, skel_idx = heapq.heappop(candidate_edges)

        if skeleton_flags[skel_idx]:
            continue


        # Paper Sec. 4.2: the cap applies to bifurcations "excluding the soma".
        # Upstream applied it to node 0 too, forcing every generated soma to have
        # at most 2 primary dendrites (real somata have 3-23).
        if len(tree.nodes[parent_id]['children']) >= (root_cap if parent_id == 0 else 2):
            continue


        tree.add_node(parent_id, skeleton[skel_idx])
        current_id = tree.current_id - 1
        visited.add(current_id)
        skeleton_flags[skel_idx] = True


        if len(visited) % 10 == 0:
            tree.update_direction()

        new_pos = skeleton[skel_idx]
        dists = distance_matrix([new_pos], skeleton)[0]
        for i, d in enumerate(dists):
            if not skeleton_flags[i] and d > 0:
                direction = skeleton[i] - new_pos
                if tree.direction is not None:
                    cos_sim = np.dot(direction, tree.direction) / (np.linalg.norm(direction) + 1e-6)
                else:
                    cos_sim = 0
                heapq.heappush(candidate_edges, (d * (gamma_main - cos_sim), current_id, i))

    swc_data = []
    for nid, node in tree.nodes.items():
        swc_entry = [
            nid,  # ID
            3 if nid != 0 else 1,  
            node['pos'][0],  # X
            node['pos'][1],  # Y
            node['pos'][2],  # Z
            soma_radius if nid == 0 else 1.0,  
            -1 if nid == 0 else node['parent'] 
        ]
        swc_data.append(swc_entry)

    return np.array(swc_data)
class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual  
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, input_channels=3, sequence_length=32):
        super().__init__()
        self.hidden_channels = 64  
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU()
        )
        self.res_layers = nn.Sequential(
            *[ResidualBlock(self.hidden_channels) for _ in range(8)]  
        )

        self.final_conv = nn.Conv1d(self.hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        # (bs, 32, 3) → (bs, 3, 32)
        x = x.transpose(1, 2)  # [bs, 3, 32]


        x = self.init_conv(x)  # [bs, 64, 32]
        x = self.res_layers(x)  # [bs, 64, 32]
        x = self.final_conv(x)  # [bs, 3, 32]

        # (bs, 3, 32) → (bs, 32, 3)
        return x.transpose(1, 2)  # [bs, 32, 3]

def generate_a_tree(neuron,model, args, radius=0.2, type_=1):
    ori_branches, offset,node_branch,branch_branch,max_dist =  neuron.easy_fetch_resample(align=args.align, move=True)
    # print('args.align:',args.align)

    branches = []
    # branch 为（32,3）ndarray

    branches = np.stack(ori_branches) # (N, 32, 3)
    branches = torch.from_numpy((branches)).float()

    max_dist, offset = np.stack(max_dist), np.stack(offset) # (521,) (521,3)
    max_dist = max_dist.reshape(offset.shape[0],1,1)
    offset = offset.reshape(max_dist.shape[0],1,offset.shape[-1])
    new_branch = model(branches).detach().numpy()
    new_branch = np.squeeze(new_branch)
    new_branch = branch_gaussian_smooth(new_branch, sigma=1.5, kernel_size=None)
    new_branch = new_branch * max_dist
    branches = new_branch + offset
    nodes = []
    node_cnt = 0
    branch_lastnode = {}
    smallnode_branch = {}
    for branch_cnt in range(len(branches)):

        branch = branches[branch_cnt]

        for cnt_32 in range(len(branch)):
            if node_cnt == 0 :
                typ = 1
                father = -2
            elif cnt_32 == 0:

                continue

            elif cnt_32 == 1:
                typ = 0

                if branch_branch[branch_cnt] != -1:
                    cu_branch = branch_cnt
                    fa_branch = branch_branch[cu_branch]
                    father = branch_lastnode[fa_branch]
                elif branch_branch[branch_cnt] == -1:
                    father =0


            else:
                typ = 0
                father = node_cnt - 1
            x = branch[cnt_32][0]
            y = branch[cnt_32][1]
            z = branch[cnt_32][2]
            node = (node_cnt+1,typ,x,y,z,1,father+1)
            nodes.append(node)
            smallnode_branch.update({node_cnt:branch_cnt})


            if cnt_32 == len(branch)-1:
                branch_lastnode.update({branch_cnt:node_cnt})
            node_cnt = node_cnt + 1

    return nodes

def load_sidecars(dataroot, cate_mids):
    """(sid, mid) -> (scale_m, centroid) from the bake sidecars.

    Raises SystemExit listing what is missing, rather than returning a partial
    map: generation is hours of GPU plus reconstruction, and a scale we cannot
    restore makes the whole run unusable downstream.
    """
    out, missing = {}, []
    for sid, mid in cate_mids:
        # not Path.with_suffix: it would eat the last dot-segment of a dotted id
        path = os.path.join(dataroot, sid, mid + '.meta.json')
        try:
            meta = json.loads(open(path).read())
            out[(sid, mid)] = (float(meta['scale_m']), [float(v) for v in meta['centroid']])
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(path)
    if missing:
        raise SystemExit(
            'cannot restore scale: {} of {} sidecars are missing or lack '
            'scale_m/centroid, e.g.\n  {}'.format(
                len(missing), len(cate_mids), '\n  '.join(missing[:5])))
    return out


# --- Stage 7: reconstruction, as a separate CPU phase -------------------------
# Reconstruction is pure-Python NumPy at ~4.2 s/neuron and is the pipeline's
# dominant cost; sampling is GPU and cheap. Upstream fuses them in one loop, so
# the GPU sits idle for ~3 CPU-hours per checkpoint on a 2,529-neuron split and a
# crash in either half discards the entire run. Splitting them also means a
# tau/gamma re-sweep costs no GPU at all -- the clouds are already on disk.
#
# These live at module level so a multiprocessing Pool can pickle them.
_AUX_MODEL = None
_RECON_KW = None


def aux_model_path(opt):
    """The auxiliary ResNet weight.

    BUGFIX: upstream pointed at './temp/resnet16model.pth', which is not in the
    release. The shipped weight is trained_model/Auxiliary.pth, resolved relative
    to this file so the CWD does not matter. Override with --aux_model.
    """
    return getattr(opt, 'aux_model', '') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'trained_model', 'Auxiliary.pth')


def load_aux_model(path):
    aux_model = ResNet18()
    aux_model.load_state_dict(torch.load(path, map_location='cpu'))
    aux_model.eval()
    return aux_model


def _recon_init(aux_path, kw):
    """Pool initialiser: load the ResNet once per worker, not once per neuron."""
    global _AUX_MODEL, _RECON_KW
    # Each worker is its own process, so leaving torch at its default thread count
    # oversubscribes the node by workers x cores and makes the pool slower than
    # serial. The work here is NumPy-bound anyway.
    torch.set_num_threads(1)
    _AUX_MODEL = load_aux_model(aux_path)
    _RECON_KW = kw


def reconstruct_one(job):
    """One generated cloud -> SWC node rows. Pure CPU: no CUDA, no dataloader.

    The fps seed is derived from the neuron's `gen_index`, not from its position
    in this worker's queue, so the output is identical whether reconstruction runs
    serially, across a pool, or resumed halfway through an interrupted job.
    """
    npy_path, seed = job
    kw = _RECON_KW

    # BUGFIX / determinism. auxi -> easy_fetch_resample -> resample_branch_by_step
    # -> farthest_point_sample_faster (utils/utils.py:28) picks its FIRST point with
    # an unseeded np.random.randint, so every branch longer than 32 points is
    # resampled from a random start. Same class of bug as the L1_medial one already
    # in the deviations table, and it is what made two calls to auxi on identical
    # input differ by ~6e-2 -- the source of the reconstruction jitter recorded as
    # an open item (mean_branch_length moving ~4e-3 between runs).
    #
    # A process-level seed hid it: morphology_gen seeds at startup, so a serial run
    # consumes one deterministic stream and reproduces itself exactly. That breaks
    # the moment reconstruction is parallel or resumed, because the stream is then
    # split differently. Seeding per neuron makes each reconstruction a pure
    # function of (cloud, seed) -- identical serial, pooled, or resumed.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    # float64, matching tools/recon_ref.py:141. The calibrated constants
    # (detect_radius 0.30, gamma_seed 0.40, tau 0.30) were fitted through that path,
    # and RECON-REF is only comparable to generation if both run the same numerics.
    # The fused upstream loop fed float32 straight off the torch tensor.
    points = np.load(npy_path).astype(np.float64)

    # BUGFIX: the generation path called L1_medial with no seed, so its centre
    # initialisation drew from the numpy GLOBAL rng and two runs on identical
    # input produced different skeletons. tools/recon_ref.py was fixed for this;
    # this call site was missed. It bites twice here: a resumed or parallel
    # reconstruction would not match a serial one, and a checkpoint sweep would
    # be measuring L1 noise alongside real differences between checkpoints.
    point_swc_L1 = L1_medial(points=points, NCenters=2048, iters=1, seed=seed)
    ske = point_swc_L1[fps(point_swc_L1, 1200, seed=seed), :]      # (1200, 3)
    connect = neuron_swc_generator(
        ske,
        detect_radius=kw['detect_radius'],
        root_cap=kw['root_cap'],
        gamma_seed=kw['gamma_seed'],
        gamma_main=kw['gamma_main'],
    )                                                              # (1200, 7)
    cut = [
        {'n': int(row[0]), 'type': int(row[1]), 'x': row[2], 'y': row[3],
         'z': row[4], 'radius': row[5], 'parent': int(row[6])}
        for row in connect
    ]
    cut = filter_short_branches(cut, length_threshold=kw['length_threshold'])
    return auxi(pd.DataFrame(cut), _AUX_MODEL)


def clouds_dir_for(opt):
    return getattr(opt, 'clouds_dir', '') or os.path.join(opt.generate_dir, 'clouds')


def sample_clouds(model, opt, gpu):
    """Phase 1 (GPU): DDPM ancestral sampling -> one .npy per neuron + an index.

    Writes unit-sphere clouds. `gen * s + m` inverts the dataset-global
    standardisation and lands back in the space detect_radius and
    length_threshold were calibrated in; no per-neuron rescale happens here or
    anywhere (see the gen_radius note below).
    """
    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
    _, test_dataloader, _, _ = get_dataloader(opt, test_dataset, test_dataset)

    def new_y_chain(device, num_chain, num_classes):
        return torch.randint(low=0, high=num_classes, size=(num_chain,), device=device)

    # Per-neuron scale, for the adapter to restore with. pc_normlize divided each
    # baked neuron by its own max radial norm and the .npy kept only the points, so
    # `scale_m` and `centroid` live in the bake sidecars. Join them by NAME here --
    # `sid`/`mid` come straight off the batch and the sidecar is exactly
    # <dataroot>/<sid>/<mid>.meta.json -- rather than by carrying a side array through
    # the dataset, whose three index spaces (pre-shuffle entries, shuffle order,
    # post-skip n_written) make an index-keyed array silently misalign. That is the
    # very failure the manifest exists to prevent.
    #
    # Preloaded up front, so a missing sidecar fails in the first second instead of
    # after hours of sampling.
    sidecars = load_sidecars(opt.dataroot, test_dataset.all_cate_mids)
    print(f'loaded {len(sidecars)} bake sidecars for scale restoration')

    cdir = clouds_dir_for(opt)
    os.makedirs(cdir, exist_ok=True)

    cap = int(getattr(opt, 'max_samples', 0) or 0)
    if cap:
        # The test loader does not shuffle and the dataset's own shuffle is seeded,
        # so the first `cap` samples are the SAME subset for every checkpoint. That
        # is the property a selection sweep needs; a random subset per checkpoint
        # would make the comparison between checkpoints meaningless.
        print(f'--max_samples {cap}: generating a fixed prefix of the split, '
              f'not the full {len(test_dataset)} neurons')

    index = []
    a = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader),
                            desc='Sampling clouds'):
            if cap and a >= cap:
                break
            x = data['test_points'].transpose(1, 2)
            m, s = data['mean'].float(), data['std'].float()
            y = data['cate_idx']
            mids, sids = data['mid'], data['sid']

            gen = model.gen_samples(x.shape, gpu,
                                    new_y_chain(gpu, y.shape[0], opt.num_classes),
                                    clip_denoised=False).detach().cpu()
            gen = gen.transpose(1, 2).contiguous()
            gen = gen * s + m                              # -> unit-sphere space

            for j, pc in enumerate(gen):
                if cap and a >= cap:
                    break
                points = pc.numpy()

                # The scale chain is already consistent, so do NOT rescale here.
                # The .npy corpus is unit-sphere by construction (pc_normlize
                # divides by the max radial norm, so max||p|| == 1 exactly), the
                # dataset standardises by a global (m, s), and the `gen * s + m`
                # above is that exact inverse -- which lands back in unit-sphere
                # space, the space detect_radius and length_threshold were
                # calibrated in (tools/recon_ref.py:154).
                #
                # Note s is ~0.23 on this corpus, so that multiply SHRINKS by
                # ~4.3x; it cannot inflate. Reconstructing before it would be
                # the actual bug: standardised space is 1/s larger than the
                # space the constants belong to.
                #
                # So a generated cloud that is not ~radius 1 is a model that has
                # not learned the training distribution -- training had zero
                # scale variance, every neuron baked to radius exactly 1. Record
                # that as a diagnostic and leave the points alone. Renormalising
                # here would hide a model failure and hand the baseline a repair
                # it did not earn.
                gen_radius = float(np.sqrt(
                    ((points - points.mean(axis=0)) ** 2).sum(axis=1)).max())

                # Name by the SOURCE neuron, not by generation order: the dataset
                # applies a deterministic shuffle (random.Random(38383)), so the
                # i-th generated sample is NOT the i-th file of the split, and
                # downstream evaluation pairs against a filename-sorted GT list.
                mid = mids[j] if isinstance(mids, (list, tuple)) else mids[j]
                sid = str(sids[j] if isinstance(sids, (list, tuple)) else sids[j])
                stem = str(mid).replace('/', '__')
                np.save(os.path.join(cdir, stem + '.npy'), points.astype(np.float32))

                gt_scale_m, gt_centroid = sidecars[(sid, str(mid))]
                index.append({
                    'file': f'{stem}.swc',
                    'cloud': f'{stem}.npy',
                    'mid': str(mid),
                    'sid': sid,
                    'cate_idx': int(y[j]),
                    # Seeds the fps draw in the reconstruction phase. Persisted so
                    # a resumed or parallel reconstruction is bit-identical to a
                    # serial one.
                    'gen_index': a,
                    # The PAIRED GT neuron's normalisation constants, i.e. S-oracle
                    # input. Named gt_* so using them for the distributional table --
                    # which plan section 5 forbids, since it scores the baseline on a
                    # quantity we handed it -- is visible at the call site.
                    'gt_scale_m': gt_scale_m,
                    'gt_centroid': gt_centroid,
                    # Max radial norm of the generated cloud. Training was
                    # unit-sphere, so a model that has learned the distribution
                    # emits ~1.0; a deviation is a training-health signal, not
                    # a scale to correct for.
                    'gen_radius': gen_radius,
                    'mean': [float(v) for v in m[j].flatten()],
                    'std': float(s[j].flatten()[0]),
                })
                a += 1

    with open(os.path.join(cdir, 'clouds.json'), 'w') as f:
        json.dump({'version': 1, 'space': 'unit_sphere', 'model': opt.model,
                   'max_samples': cap, 'neurons': index}, f, indent=1)
    print(f'wrote {len(index)} clouds + clouds.json to {cdir}')
    report_gen_radius(index)
    return index


def report_gen_radius(index):
    """Cheapest available read on whether the samples are in-distribution at all."""
    if not index:
        return
    radii = np.array([r['gen_radius'] for r in index])
    print('generated radius (training distribution is exactly 1.0): '
          'median {:.3g}  p10 {:.3g}  p90 {:.3g}'.format(
              float(np.median(radii)), float(np.percentile(radii, 10)),
              float(np.percentile(radii, 90))))
    if not 0.5 < float(np.median(radii)) < 2.0:
        warnings.warn(
            'generated clouds have median radius {:.3g} against a training '
            'distribution of exactly 1.0 -- the samples are far out of '
            'distribution, so detect_radius and length_threshold are being '
            'applied at the wrong scale and the morphometrics are not '
            'meaningful. Train longer; do not rescale here.'.format(
                float(np.median(radii))), RuntimeWarning)


def reconstruct_clouds(opt, outf_syn):
    """Phase 2 (CPU, resumable, parallel): clouds -> SWC + the v2 manifest.

    Resumable at neuron granularity: a neuron whose .swc is already on disk is
    skipped, so an interrupted job restarts where it stopped rather than redoing
    hours of work. The manifest is rebuilt in full from clouds.json every time, so
    a resumed run still emits a complete bijection onto the split.
    """
    cdir = clouds_dir_for(opt)
    idx_path = os.path.join(cdir, 'clouds.json')
    if not os.path.isfile(idx_path):
        raise SystemExit(
            f'no {idx_path}; run --stage sample first (or --stage both).')
    doc = json.load(open(idx_path))
    index = doc['neurons']
    if doc.get('space') != 'unit_sphere':
        raise SystemExit(f'{idx_path} declares space={doc.get("space")!r}; '
                         'reconstruction is only calibrated for unit_sphere.')

    os.makedirs(opt.generate_dir, exist_ok=True)
    kw = {'detect_radius': getattr(opt, 'detect_radius', None),
          'root_cap': getattr(opt, 'root_cap', 2),
          'gamma_seed': getattr(opt, 'gamma_seed', 1.0),
          'gamma_main': getattr(opt, 'gamma_main', 1.2),
          'length_threshold': getattr(opt, 'length_threshold', 0.1)}
    print('reconstruction settings: %s' % kw)

    todo, done = [], 0
    for row in index:
        dest = os.path.join(opt.generate_dir, row['file'])
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            done += 1
            continue
        todo.append((row, (os.path.join(cdir, row['cloud']),
                           int(opt.manualSeed) + int(row['gen_index']))))
    if done:
        print(f'resuming: {done}/{len(index)} already reconstructed, {len(todo)} to go')

    aux_path = aux_model_path(opt)
    nproc = int(getattr(opt, 'recon_workers', 0) or 0)

    def _write(row, nodes):
        with open(os.path.join(opt.generate_dir, row['file']), 'w') as f:
            for node in nodes:
                f.write(' '.join(map(str, node)) + '\n')

    if nproc > 1 and todo:
        import multiprocessing as _mp
        # 'spawn' rather than fork: the parent may hold a CUDA context from the
        # sampling phase, and forking one is undefined behaviour.
        ctx = _mp.get_context('spawn')
        with ctx.Pool(nproc, initializer=_recon_init,
                      initargs=(aux_path, kw)) as pool:
            for row, nodes in tqdm(
                    zip([r for r, _ in todo],
                        pool.imap(reconstruct_one, [j for _, j in todo], chunksize=1)),
                    total=len(todo), desc=f'Reconstructing ({nproc} workers)'):
                _write(row, nodes)
    else:
        _recon_init(aux_path, kw)
        for row, job in tqdm(todo, desc='Reconstructing (serial)'):
            _write(row, reconstruct_one(job))

    # Explicit generated -> source join for the evaluation harness.
    # Envelope, not a bare list: the SWCs on disk are in UNIT-SPHERE coordinates
    # (reconstruction runs there, and scale is restored downstream), so say so
    # explicitly and give the adapter something to assert on rather than infer.
    manifest = [{k: v for k, v in row.items() if k != 'cloud'} for row in index]
    with open(os.path.join(opt.generate_dir, 'manifest.json'), 'w') as f:
        json.dump({'version': 2,
                   'space': 'unit_sphere',
                   'scale_restored': False,
                   'neurons': manifest}, f, indent=1)
    print(f'wrote {len(manifest)} neurons + manifest.json to {opt.generate_dir}')
    report_gen_radius(manifest)
    return outf_syn


def generate_eval(model, opt, gpu, outf_syn, evaluator):
    stage = getattr(opt, 'stage', 'both')
    if stage in ('sample', 'both'):
        sample_clouds(model, opt, gpu)
    if stage == 'sample':
        print('--stage sample: clouds written; run --stage reconstruct next.')
        return outf_syn
    return reconstruct_clouds(opt, outf_syn)



def _require_dataroot(opt):
    """Fail loudly and early on an unset/missing dataroot.

    The loader silently skips files it cannot read, so a wrong path yields an empty
    dataset and a training loop that appears to run while doing nothing.
    """
    import sys as _sys
    from pathlib import Path as _P
    if not opt.dataroot:
        _sys.exit('--dataroot is not set. Pass it, or export MORPHOGEN_DATAROOT=/path/to/npy_root')
    for cate in str(opt.category).split(','):
        d = _P(opt.dataroot) / cate.strip()
        if not d.is_dir():
            _sys.exit(f'no such synset directory: {d}\n'
                      f'  expected <dataroot>/<category>/{{train,val,test}}/*.npy\n'
                      f'  dataroot={opt.dataroot!r} category={opt.category!r}')
        if not list((d / 'train').glob('*.npy')) and not list((d / 'val').glob('*.npy')):
            _sys.exit(f'{d} contains no .npy under train/ or val/ '
                      '-- run tools/swc_to_morphogen_npy.py first')

def main(opt):
    # The reconstruction phase reads clouds off disk and never opens the corpus,
    # so it must not demand a dataroot -- it is meant to run on a CPU node that
    # may not even have the baked .npy staged.
    if getattr(opt, 'stage', 'both') != 'reconstruct':
        _require_dataroot(opt)
    output_dir = get_output_dir(opt.generate_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    opt.dist_url = f'tcp://{opt.node}:{opt.port}'
    print('Using url {}'.format(opt.dist_url))

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        test(opt.gpu, opt, output_dir)

def test(gpu, opt, output_dir):

    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    # --stage reconstruct is pure CPU: it reads clouds off disk and never touches
    # the DDPM. Returning here means the reconstruction job needs no GPU
    # allocation and no checkpoint -- which is the whole point of the split, since
    # it is the long half and the cluster's CPU nodes are far easier to get.
    if getattr(opt, 'stage', 'both') == 'reconstruct':
        return reconstruct_clouds(opt, outf_syn)

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        # 'single' is the default, and it called .cuda() unconditionally -- so
        # generation died with "Torch not compiled with CUDA enabled" on any
        # CPU-only machine, before a single sample. The model is already on the
        # right device from the model.to(device) above; DataParallel over CPU is
        # meaningless, so on CPU just leave it alone. The EMA key remap below
        # already handles both the wrapped and unwrapped layouts.
        if torch.cuda.is_available():
            def _transform_(m):
                return nn.parallel.DataParallel(m)
            model = model.cuda()
            model.multi_gpu_wrapper(_transform_)
        else:
            print('CUDA unavailable: running on CPU, no DataParallel wrapper')

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

        logger.info("Model = %s" % str(model))
        total_params = sum(param.numel() for param in model.parameters())/1e6
        logger.info("Total_params = %s MB " % str(total_params))    # S4: 32.81 MB

    model.eval()

    evaluator = Evaluator(results_dir=output_dir)    

    with torch.no_grad():
        
        if should_diag:
            logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model, map_location='cpu')

        # Generate from the EMA weights whenever the checkpoint carries them.
        # The shipped code always took 'model_state' (the raw weights), so a run
        # that trained with --use_ema and selected a checkpoint on EMA would
        # then have generated from something else entirely, with nothing in the
        # output to reveal it.
        #
        # Key prefixes differ: the EMA copy is deepcopy'd before
        # multi_gpu_wrapper runs, so it is stored unwrapped ('model.*') while
        # 'model_state' and the live model here are wrapped ('model.module.*').
        # Remap to whatever this model actually asks for.
        source = 'ema' if 'ema' in resumed_param else 'model_state'
        state = resumed_param[source]

        wants_module = any(k.startswith('model.module.')
                           for k in model.state_dict())

        def _remap(k):
            if wants_module and k.startswith('model.') and not k.startswith('model.module.'):
                return 'model.module.' + k[len('model.'):]
            if not wants_module and k.startswith('model.module.'):
                return 'model.' + k[len('model.module.'):]
            return k

        if should_diag:
            logger.info('Loading weights from %s[%s]' % (opt.model, source))
        model.load_state_dict({_remap(k): v for k, v in state.items()})

        opt.eval_path = os.path.join(outf_syn, 'samples.pth')
        Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
        
        stats = generate_eval(model, opt, gpu, outf_syn, evaluator)

        if should_diag:
            logger.info(stats)
        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')
    
    '''eval'''
    parser.add_argument('--eval_path',default='')
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    # --- reconstruction controls (see neuron_swc_generator docstring) ---
    parser.add_argument('--detect_radius', default=None, type=float,
                        help='soma density-detection radius in NORMALIZED units. The shipped '
                             'code reused --soma_radius (5.0 um) here, which exceeds the unit-sphere '
                             'extent and makes every density tie, so the root became arbitrary. '
                             'Use 5.0/m where m is the max radial norm before pc_normlize '
                             '(~0.02 for the MICrONS cortical corpus).')
    parser.add_argument('--root_cap', default=2, type=int,
                        help='max children allowed on the soma. Paper Sec. 4.2 exempts the soma '
                             'from the 2-child rule; the released code did not. Set e.g. 23 to '
                             'restore the documented behaviour.')
    parser.add_argument('--length_threshold', default=0.1, type=float,
                        help='filter_short_branches threshold, in NORMALIZED units.')
    parser.add_argument('--aux_model', default='', type=str,
                        help='auxiliary CNN checkpoint (default: trained_model/Auxiliary.pth)')
    parser.add_argument('--gamma_seed', default=1.0, type=float,
                        help='directional coefficient for SOMA edges. Shipped value 1.0 leaves '
                             'soma edges uncompetitive against the discounted frontier edges, so '
                             'root degree collapses to 1-2 (GT median 7). Calibrated on train, '
                             '0.40 matches the GT median stem count and basal fraction '
                             '-- the MorphoGen+ arm.')
    # --- Stage 7: sampling and reconstruction are separate phases ------------
    # Reconstruction is pure-Python CPU at ~4.2 s/neuron and sampling is GPU, so
    # fusing them (as upstream does) idles the GPU for ~3 hours per checkpoint on
    # a 2,529-neuron split and loses the whole run if either half dies. The
    # clouds are persisted between the phases, which also means a tau/gamma
    # re-sweep costs no GPU at all.
    parser.add_argument('--stage', default='both',
                        choices=['sample', 'reconstruct', 'both'],
                        help="'sample' = DDPM only (GPU, writes clouds/); "
                             "'reconstruct' = clouds/ -> SWC (CPU, resumable); "
                             "'both' = one after the other.")
    parser.add_argument('--clouds_dir', default='',
                        help='where the raw generated point clouds live. '
                             'Defaults to <generate_dir>/clouds.')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='cap the number of neurons generated (0 = the whole '
                             'split). The dataloader order is fixed, so a cap '
                             'yields the SAME subset for every checkpoint -- which '
                             'is what makes a selection sweep comparable.')
    parser.add_argument('--recon_workers', type=int, default=0,
                        help='processes for the reconstruction phase (0 = serial).')
    parser.add_argument('--gamma_main', default=1.2, type=float,
                        help="frontier-edge directional coefficient; the paper's stated gamma.")
    parser.add_argument('--model_dir', type=str, default=r'/mnt/d/hyzhou/point cloud/temp', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ct', help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--category', default=os.environ.get('MORPHOGEN_CATEGORY', 'neurons'),
                        help="synset name(s); must match what was used for training")
    parser.add_argument('--bs', type=int, default=2, help='input batch size')
    parser.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-S/4")
    parser.add_argument("--voxel_size", type=int, choices=[16, 32, 64], default=32)
# Data paths default from the environment so a forgotten flag fails loudly instead of
# silently falling back to the original authors' cluster layout. Set once:
#   export MORPHOGEN_DATAROOT=/scratch/$USER/morphogen_npy
#   export MORPHOGEN_RUNS=/scratch/$USER/mg_runs
#   export MORPHOGEN_GENDIR=/scratch/$USER/mg_gen
    parser.add_argument('--model', default='', help='checkpoint to generate from (.pth)')
    parser.add_argument('--generate_dir', default=os.environ.get('MORPHOGEN_GENDIR', './generated'),
                        help='where generated SWCs + manifest.json go (env: MORPHOGEN_GENDIR)')
    parser.add_argument('--dataroot', default=os.environ.get('MORPHOGEN_DATAROOT', ''),
                        help='root holding <synset>/{train,val,test}/*.npy (env: MORPHOGEN_DATAROOT)')
    opt = parser.parse_args()


    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
