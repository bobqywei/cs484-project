import torch


def to_device(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = to_device(obj[k], device)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = to_device(obj[i], device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    return obj
