import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class BaseLoss(object):
    def __init__(self, config):
        self.config = config
        self.H = config['img_hgt']
        self.W = config['img_wid']
        self.bsize = config['batch_size']

        x, y = np.meshgrid(range(self.W), range(self.H))
        xy_coords = np.stack([x, y], axis=0).astype(np.float32)
        xy_coords = torch.from_numpy(xy_coords).view(1, 2,-1).to(config['device']) # [1,2,H*W]
        xy_coords = xy_coords.repeat(self.bsize, 1, 1) # [B,2,H*W]

        self.ones = torch.ones(self.bsize, 1, self.H * self.W, device=config['device'])
        # 2D image plane coords to homogeneous 3D coords
        self.img_coords = torch.cat([xy_coords, self.ones], dim=1) # [B,3,H*W]

        self.SSIMLoss = SSIM()

    def _elemwise_reprojection_loss(self, pred, target):
        l1_loss = torch.abs(pred - target).mean(dim=1, keepdim=True)
        ssim_loss = self.SSIMLoss(pred, target).mean(dim=1, keepdim=True)
        return ssim_loss*0.85 + l1_loss*0.15

    # t_img, tPrime_img: [B,3,H,W]
    # Ks: 4 x [B,4,4]
    # invKs: 4 x [B,4,4]
    # pose_preds (for t -> t'): [B,6]
    # disparity_preds (for t): 4 x [B,scaleH,scaleW]
    def compute_proj_loss(self, t_img, tPrime_img, Ks, invKs, pose_preds, disparity_preds, invert=False):
        bsize = t_img.shape[0]

        R = rotation_matrix(pose_preds[:,:3])
        T = translation_matrix(pose_preds[:,3:])
        P = torch.matmul(R.transpose(1,2), T*-1) if invert else torch.matmul(T, R) # [B,4,4]

        proj_loss_across_scales = []

        for i in range(self.config['num_scales']):
            depth = disparity_to_depth(
                disparity_preds[i], self.config['min_depth'], self.config['max_depth'])
            # Upsample all depth maps to original [B,H,W]
            if i > 0:
                depth = F.interpolate(depth, (self.H, self.W), mode="bilinear", align_corners=False)

            K = Ks[i]
            invK = invKs[i]

            # project from camera t normalized image coords to coords w.r.t. camera t'
            cam_t_coords = torch.matmul(invK[:,:3,:3], self.img_coords) # [B,3,H*W]
            # gives us (d*x, d*y, d, 1) homogeneous coords
            cam_t_coords = depth.view(bsize, 1, -1) * cam_t_coords
            cam_t_coords = torch.cat([cam_t_coords, self.ones], dim=1) # [B,4,H*W]
            
            # projection matrix to camera t' normalized image coords 
            # don't care about 4th coordinate in homogeneous representation
            proj_tPrime = torch.matmul(K, P)[:,:3,:]
            tPrime_coords = torch.matmul(proj_tPrime, cam_t_coords) # [B,3,H*W]

            # homogeneous coords to xy coords
            tPrime_coords = tPrime_coords[:,:2,:] / (tPrime_coords[:,2:3,:] + 1e-10)
            tPrime_coords = tPrime_coords.view(bsize, 2, self.H, self.W).permute(0, 2, 3, 1) # [B,H,W,2]
            # normalize t' image coords to [-1, 1]
            tPrime_coords[:,:,:,0] /= self.W - 1
            tPrime_coords[:,:,:,1] /= self.H - 1
            tPrime_coords = (tPrime_coords - 0.5) * 2
            
            # form the predicted rgb image for t'
            pred_tPrime_img = F.grid_sample(t_img, tPrime_coords, padding_mode="border")

            # compute the reprojection loss
            reproj_loss = self._elemwise_reprojection_loss(pred_tPrime_img, tPrime_img) # [B,1,H,W]
            # use automasking for removing stationary pixels from loss
            # takes the minimum between reproj loss and unwarped loss at each pixel
            if self.config['use_mask']:
                unwarped_loss = self._elemwise_reprojection_loss(t_img, tPrime_img)
                reproj_loss, _ = torch.cat([reproj_loss, unwarped_loss], dim=1).min(dim=1, keepdim=True)
            
            proj_loss_across_scales.append(reproj_loss)

        # mean across scales, final shape of [B,1,H,W]
        proj_loss = torch.cat(proj_loss_across_scales, dim=1).mean(dim=1, keepdim=True)
        return proj_loss


    # disp_imgs (for t): 4 x [B,scaleH,scaleW]
    # rgb_img (for t): [B,3,H,W] 
    def compute_smooth_loss(self, disp_imgs, rgb_img):
        # using rgb image for simple object edge awareness
        grad_rgb_x = torch.mean(torch.abs(rgb_img[:, :, :, :-1] - rgb_img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_rgb_y = torch.mean(torch.abs(rgb_img[:, :, :-1, :] - rgb_img[:, :, 1:, :]), dim=1, keepdim=True)
        scale_x = torch.exp(-grad_rgb_x)
        scale_y = torch.exp(-grad_rgb_y)

        total_smooth_loss = 0
        for scale in range(self.config['num_scales']):
            disp_img = disp_imgs[scale]
            if scale > 0:
                disp_img = F.interpolate(
                    disp_img, (self.H, self.W), mode="bilinear", align_corners=False)
                
            # normalize by the mean disparity
            norm_disp = disp_img / (disp_img.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + 1e-10)

            grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
            grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])
            grad_disp_x *= scale_x
            grad_disp_y *= scale_y

            smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
            # total_smooth_loss += smooth_loss / (2 ** scale)
            total_smooth_loss += smooth_loss

        return self.config['smooth_weight'] * total_smooth_loss / self.config['num_scales']
