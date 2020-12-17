import os
import sys
import yaml

from torch.utils.data.dataloader import DataLoader
from dataset import KITTIDatasetTrain

if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 2
    with open(os.path.join(os.curdir, args[1]), 'r') as f:
        config = yaml.safe_load(f)

    train_dataset = KITTIDatasetTrain('data/train.txt', config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    
    # val_dataset = KITTIDatasetTrain('data/val.txt', config)
    # val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    