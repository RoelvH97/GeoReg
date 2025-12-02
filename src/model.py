# import necessary libraries
import cv2
import hydra
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time
import torch
import torch.nn as nn

from diffdrr.data import read
from diffdrr.drr import convert, DRR
from diffdrr.metrics import *
from functools import wraps
from monai.losses.dice import *
from pathlib import Path
from pytorch3d.transforms import so3_log_map
from scipy.ndimage import label
from skimage.transform import resize
from torch.optim import *
from torch.optim.lr_scheduler import *
from .data import sitk_to_numpy

# try to import bilateral_filter_layer, fall back to cv2 if not available
try:
    from bilateral_filter_layer import BilateralFilter3d
    HAS_BILATERAL_LAYER = True
except ImportError:
    HAS_BILATERAL_LAYER = False


def time_it(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"{func.__name__} completed in {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}")
        return result
    return wrapper


class BiplanarReg(nn.Module):
    def __init__(self, config, id_dict):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize bilateral filter layer if available
        if HAS_BILATERAL_LAYER:
            self.layer = BilateralFilter3d(*config.sigmas, use_gpu=torch.cuda.is_available()).to(self.device)
        else:
            self.layer = None

        # create output dir
        base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        id_ = Path(id_dict['CTA']).stem.split('_')[0]
        self.output_dir = base_dir / id_ / config.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_dir / 'images'
        self.image_dir.mkdir(exist_ok=True)

        # load data
        paths = self._get_dsa_paths(config.timestamp, id_dict)
        cta = read(id_dict['CTA'], id_dict['CTA_mask'], labels=[0, 1])
        self.dsa, self.dsa_mask = self._prepare_dsa_data(paths)

        self.dists = {}
        self.drrs = {}
        self.params = nn.ParameterDict()
        for v in ['lat', 'pa']:
            with open(paths[2][v]) as f:
                metadata = json.load(f)

            self.dists[v] = metadata['d_source_to_detector'] - config.detector_dist
            self.drrs[v] = DRR(cta, sdd=metadata['d_source_to_detector'],
                               height=config.detector_size[0], delx=config.detector_spacing[0],
                               stop_gradients_through_grid_sample=True).to(self.device)
            self.params[v] = self._initialize_params(v)

        # loss
        self.criterion = eval(config.criterion_img)(**(config.criterion_img_kwargs or {}))
        self.dice_loss = eval(config.criterion_msk)()

        # setup optimizer
        self.current_iter = 0
        self.optimizer = eval(config.optimizer)(self.parameters(), **(self.config.optimizer_kwargs or {}))
        self.scheduler = eval(config.scheduler)(self.optimizer, **(config.scheduler_kwargs or {}))
        

    def _get_dsa_paths(self, timestamp, id_dict):
        """Get DSA and camera file paths based on DSA type."""
        suffix_map = {"pre": ["_a", "_b"],
                      "post": ["_c", "_d"]}

        suffixes = suffix_map['pre'] if 'pre' in timestamp else suffix_map['post']
        img_paths = {'lat': str(id_dict['DSA']) + suffixes[0] + '_0000.nii.gz',
                     'pa': str(id_dict['DSA']) + suffixes[1] + '_0000.nii.gz'}
        msk_paths = {'lat': str(id_dict['DSA_mask']) + suffixes[0] + '.nii.gz',
                     'pa': str(id_dict['DSA_mask']) + suffixes[1] + '.nii.gz'}
        meta_paths = {'lat': str(id_dict['DSA_metadata']) + suffixes[0] + '.json',
                      'pa': str(id_dict['DSA_metadata']) + suffixes[1] + '.json'}
        return img_paths, msk_paths, meta_paths
    
    def _prepare_dsa_data(self, paths):         
        imgs = {}
        msks = {}
   
        for v in ['lat', 'pa']:
            # prepare image
            img, _, _ = sitk_to_numpy(paths[0][v])
            img = np.max(img, axis=0)

            # calculate metrics over valid region
            img_nonzero = np.sum(img, axis=0) != 0
            img_valid = img[:, img_nonzero]
            vmin, vmax = np.percentile(img_valid, [25, 75])
            if vmin == vmax:
                vmax += 40

            msk, _, _ = sitk_to_numpy(paths[1][v])

            # get largest mask component
            labeled, _ = label(msk[0])
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            msk = (labeled == largest_component).astype(float)

            # norm
            img = np.clip(img, vmin, vmax)
            img -= np.min(img)

            # reshape
            if img.shape != self.config.detector_size:
                img = resize(img, self.config.detector_size, preserve_range=True, anti_aliasing=True)
                msk = resize(msk, self.config.detector_size, preserve_range=True, anti_aliasing=False)

            # filter - use bilateral_filter_layer if available, otherwise cv2
            if HAS_BILATERAL_LAYER and self.layer is not None:
                img = torch.tensor(img, device=self.device, dtype=torch.float32)[None, None, None]
                msk_tensor = torch.tensor(msk, device=self.device, dtype=torch.float32)[None, None]
                with torch.no_grad():
                    img = self.layer(img)[:, :, 0] * msk_tensor
                imgs[v] = img
                msks[v] = msk_tensor
            else:
                img = cv2.bilateralFilter(img.astype(np.float32), *self.config.sigmas)
                img *= msk
                imgs[v] = torch.tensor(img, device=self.device, dtype=torch.float32)[None, None]
                msks[v] = torch.tensor(msk, device=self.device, dtype=torch.float32)[None, None]

        return imgs, msks

    def _initialize_params(self, view):    
        rot_init = [[90, 0, 0]] if view == 'lat' else [[0, 0, 0]]    
        tra_init = [[0, self.dists[view], 0]]

        rot_init = torch.tensor(rot_init, device=self.device, dtype=torch.float32) / 180 * torch.pi
        tra_init = torch.tensor(tra_init, device=self.device, dtype=torch.float32) / self.config.multiplier

        return nn.ParameterList([nn.Parameter(rot_init), nn.Parameter(tra_init)])

    def _extract_parameters(self, mode, to_cpu=True, to_list=True):
        """Extract rotation and translation parameters as lists."""
        rot = self.params[mode][0]
        tra = self.params[mode][1] * self.config.multiplier

        if to_cpu:
            rot, tra = rot.detach().cpu(), tra.detach().cpu()
        if to_list:
            rot, tra = rot.tolist(), tra.tolist()
        return rot, tra
    
    def _geodesic_distance_so3(self):
        rots = []
        for v in ['lat', 'pa']:
            rot, tra = self._extract_parameters(v, to_cpu=False, to_list=False)
            pose = convert(rot, tra, parameterization='euler_angles', convention='ZYX')
            rots.append(pose.matrix[0, :3, :3])

        # calculate geodesic distance
        R_rel = rots[0].T @ rots[1]
        log_R_rel = so3_log_map(R_rel[None])

        return torch.norm(log_R_rel[0])
    
    def _compute_loss(self):
        ncc_losses = []
        dsc_losses = []

        for v in ['lat', 'pa']:
            rot, tra = self._extract_parameters(v, False, False)
            estimate = self.drrs[v](rot, tra, parameterization='euler_angles',
                                    convention='ZYX', mask_to_channels=True)
        
            # ncc loss
            ncc_losses.append(-self.criterion(self.dsa[v], estimate.sum(dim=1, keepdim=True)))

            # dice loss
            msk_target = (self.dsa_mask[v] > 0).float()
            estimate_pred = torch.sigmoid(estimate[:, 1:2])
            dsc_losses.append(self.dice_loss(estimate_pred, msk_target))

        return ncc_losses, dsc_losses

    def _plot(self, ncc_losses, dsc_losses):
        """Save comparison plot of ground truth DSA and current estimates."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for idx, v in enumerate(['lat', 'pa']):
            # compute estimate
            rot, tra = self._extract_parameters(v, to_cpu=False, to_list=False)
            estimate = self.drrs[v](rot, tra, parameterization='euler_angles',
                                    convention='ZYX', mask_to_channels=True)
            estimate = estimate.sum(dim=1, keepdim=True)

            # plot ground truth DSA
            dsa_np = self.dsa[v].squeeze().detach().cpu().numpy()
            axes[0, idx].imshow(dsa_np, cmap='gray')
            axes[0, idx].set_title(f'DSA {v.upper()}')
            axes[0, idx].axis('off')

            # plot current estimate
            estimate_np = estimate.squeeze().detach().cpu().numpy()
            axes[1, idx].imshow(estimate_np, cmap='gray')
            axes[1, idx].set_title(f'Estimate {v.upper()} (NCC: {ncc_losses[idx].item():.4f}, DSC: {dsc_losses[idx].item():.4f})')
            axes[1, idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.image_dir / f'iteration_{self.current_iter:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_video_from_images(self, fps=30):
        """Create MP4 video from saved iteration images and delete the images."""
        # get all image files sorted by iteration number
        image_files = sorted(self.image_dir.glob('iteration_*.png'))

        # read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        height, width, _ = first_img.shape

        # create video writer
        video_path = self.output_dir / 'optimization_progress.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # write all images to video
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            video.write(img)
        video.release()

        # delete the image folder and all its contents
        shutil.rmtree(self.image_dir)
        logging.info(f"Video saved to {video_path}")

    @time_it
    def fit(self):
        ncc = {'lat': [], 'pa': []}
        dsc = {'lat': [], 'pa': []}
        params = {'lat': {'rot': [], 'tra': []},
                  'pa': {'rot': [], 'tra': []}}
        
        # initial loss
        ncc_losses, dsc_losses = self._compute_loss()
        for i, v in enumerate(['lat', 'pa']):
            ncc[v].append(ncc_losses[i].item())
            dsc[v].append(dsc_losses[i].item())

        # track best
        ncc_best = {v: ncc[v][0] for v in ['lat', 'pa']}
        dsc_best = {v: dsc[v][0] for v in ['lat', 'pa']}
        rot_best = {v: self._extract_parameters(v)[0] for v in ['lat', 'pa']}
        tra_best = {v: self._extract_parameters(v)[1] for v in ['lat', 'pa']}
        for i in range(1, self.config.num_iter + 1):
            self.current_iter = i

            for v in ['lat', 'pa']:
                rot, tra = self._extract_parameters(v)
                params[v]['rot'].append(rot)
                params[v]['tra'].append(tra)

            if i == 1:
                logging.info(f"Initial parameters:")
                for v in ['lat', 'pa']:
                    logging.info(f"  {v.upper()}:")
                    logging.info(f"    NCC initial: {ncc_best[v]:.4f}")
                    logging.info(f"    DSC initial: {dsc_best[v]:.4f}")
                    logging.info(f"    Rotation (alpha, beta, gamma): {rot_best[v]}")
                    logging.info(f"    Translation (bx, by, bz): {tra_best[v]}")

            # optimization step
            self.optimizer.zero_grad()

            ncc_losses, dsc_losses = self._compute_loss()
            loss = 0.0
            for ncc_loss, dsc_loss in zip(ncc_losses, dsc_losses):
                loss += self.config.alpha * ncc_loss + (1 - self.config.alpha) * dsc_loss
            if self.config.lambda_so3 > 0:
                reg_so3 = torch.abs(self._geodesic_distance_so3() - (torch.pi / 2))
                loss += self.config.lambda_so3 * reg_so3

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # plot
            self._plot(ncc_losses, dsc_losses)

            # log
            for j, v in enumerate(['lat', 'pa']):
                ncc[v].append(ncc_losses[j].item())
                dsc[v].append(dsc_losses[j].item())

                rot, tra = self._extract_parameters(v)
                params[v]['rot'].append(rot)
                params[v]['tra'].append(tra)

                if ncc_losses[j].item() < ncc_best[v]:
                    ncc_best[v] = ncc_losses[j].item()
                    dsc_best[v] = dsc_losses[j].item()
                    rot_best[v], tra_best[v] = self._extract_parameters(v)
                    logging.info(f"Iter {i}: New best NCC for {v.upper()}: {ncc_best[v]:.4f}")
                    logging.info(f"         DSC: {dsc_best[v]:.4f}")

        # save best parameters to JSON
        best_params = {
            'ncc_best': ncc_best,
            'dsc_best': dsc_best,
            'rot_best': rot_best,
            'tra_best': tra_best
        }
        with open(self.output_dir / 'best_parameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        # create video from images and delete PNGs
        self._create_video_from_images()
