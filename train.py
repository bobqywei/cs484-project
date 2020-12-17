import os
import torch

from tqdm import tqdm
from utils import to_device


def get_losses_over_frames(posenet, data, disparity_imgs, config):
    reproj_loss_all_frames = []
    smooth_loss_all_frames = 0

    for tPrime_k in ['prev', 'next']:
        img_pair = torch.concat(
            [
                data['img_aug'] if tPrime_k == 'next' else data['prev_img_aug'],
                data['next_img_aug'] if tPrime_k == 'next' else data['img_aug'] 
            ],
            dim=1
        ) # [B,6,H,W]
        pose_params = posenet(img_pair) # [B,6]

        # pixel_wise reprojection loss: [B,1,H,W]
        reproj_loss = config['Loss'].compute_proj_loss(
            t_img=data['img'] if tPrime_k == 'next' else data['prev_img'],
            tPrime_img=data['next_img'] if tPrime_k == 'next' else data['img'],
            Ks=data['K'],
            invKs=data['invK'],
            pose_preds=pose_params,
            disparity_preds=disparity_imgs,
            invert=(tPrime_k == 'prev'))
        reproj_loss_all_frames.append(reproj_loss)
        
        # mean smooth loss across all elements
        smooth_loss_all_frames += config['Loss'].compute_smooth_loss(disparity_imgs, data['img'])

    # mean smooth loss over both frames
    smooth_loss_all_frames /= 2.0

    # take minimum reprojection loss across frames
    if config['min_proj']:
        reproj_loss_all_frames = torch.cat(reproj_loss_all_frames, dim=1).min(dim=1)
    else:
        reproj_loss_all_frames = torch.cat(reproj_loss_all_frames, dim=1).mean(dim=1)
    reproj_loss_all_frames = reproj_loss_all_frames.mean()

    return reproj_loss_all_frames, smooth_loss_all_frames


def train_one_epoch(
    epoch, 
    depthnet, 
    posenet,
    optimizer, 
    train_dataloader,
    tb_logger, 
    config):
    
    print('\nEPOCH #{}'.format(epoch))

    for i, data in enumerate(train_dataloader, start=1):
        bsize = data['img'].shape[0]
        data = to_device(data, config['device'])
        disparity_imgs = depthnet(data['img_aug']) # 4 x [B,scaleH,scaleW]
        
        # Compute losses over 
        reproj_loss, smooth_loss = get_losses_over_frames(posenet, data, disparity_imgs, config)
        loss = reproj_loss + smooth_loss
        
        # Backpropagate total losses
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if i % config['log_freq'] == 0:
            reproj_loss_scalar = reproj_loss.item()
            smooth_loss_scalar = smooth_loss.item()
            loss_scalar = reproj_loss_scalar + smooth_loss_scalar

            print('{}/{} - [{}]: reproj={:.4f}, smooth={:.4f}, total={:.4f}'.format(
                i, len(train_dataloader), epoch, reproj_loss_scalar, smooth_loss_scalar, loss_scalar))
            
            total_iter = (i-1)*bsize + epoch * len(train_dataloader.dataset)
            tb_logger.add_scalar('train/total_loss', loss_scalar, total_iter)
            tb_logger.add_scalar('train/reproj_loss', reproj_loss_scalar, total_iter)
            tb_logger.add_scalar('train/smooth_loss', smooth_loss_scalar, total_iter)


def validate(epoch, depthnet, posenet, val_dataloader, tb_logger, config):
    print('Validating...')
    with torch.no_grad():
        total_reproj_loss = 0.0
        total_smooth_loss = 0.0
        
        for data in tqdm(val_dataloader):
            data = to_device(data, config['device'])
            disparity_imgs = depthnet(data['img_aug']) # 4 x [B,scaleH,scaleW]
            
            reproj_loss, smooth_loss = get_losses_over_frames(posenet, data, disparity_imgs, config)
            total_reproj_loss += reproj_loss.item()
            total_smooth_loss += smooth_loss.item()

        total_reproj_loss /= len(val_dataloader)
        total_smooth_loss /= len(val_dataloader)
        total_loss = total_reproj_loss + total_smooth_loss
        
        # Logging
        print('reproj={:.4f}, smooth={:.4f}, total={:.4f}'.format(total_reproj_loss, total_smooth_loss, total_loss))
        tb_logger.add_scalar('val/total_loss', total_loss, epoch)
        tb_logger.add_scalar('val/reproj_loss', total_reproj_loss, epoch)
        tb_logger.add_scalar('val/smooth_loss', total_smooth_loss, epoch)
