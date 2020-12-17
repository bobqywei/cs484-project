import os
import sys
import yaml
import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import KITTIDatasetTrain
from train import train_one_epoch, validate
from loss import BaseLoss
from posenet import PoseNet
from depthnet import DepthNet


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['device'] = torch.device('cuda' if config['cuda'] else 'cpu')

    depthnet = DepthNet(config).to(config['device'])

    if not config['eval']:
        ckpt_dir = os.path.join('checkpoints', config['exp_name'])
        tb_dir = os.path.join('tensorboards', config['exp_name'])
        for path in [ckpt_dir, tb_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        tb_logger = SummaryWriter(log_dir=tb_dir)

        train_dataset = KITTIDatasetTrain('data/train.txt', config)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'], 
            pin_memory=True,
            drop_last=True)
        
        val_dataset = KITTIDatasetTrain('data/val.txt', config)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True)

        posenet = PoseNet(config).to(config['device'])

        config['Loss'] = BaseLoss(config)

        params = list(posenet.parameters()) + list(depthnet.parameters())
        optimizer = torch.optim.Adam(params, config['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['schedule'], gamma=0.1)
        
        for epoch in range(1, config['num_epochs']+1):
            # train single epoch
            depthnet.train()
            posenet.train()
            train_one_epoch(epoch, depthnet, posenet, optimizer, train_dataloader, tb_logger, config)
            # save depth model
            torch.save(depthnet.state_dict(), os.path.join(ckpt_dir, 'depthnet_{}.pth'.format(epoch)))
            
            # validate
            depthnet.eval()
            posenet.eval()
            validate(epoch, depthnet, posenet, val_dataloader, tb_logger, config)
            
            # step learning rate scheduler
            scheduler.step()


if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 2
    main(os.path.join(os.curdir, args[1]))
    