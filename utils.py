import torch
import torch.nn as nn

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


def disparity_to_depth(disp, min_depth, max_depth):
    max_disp = 1 / min_depth
    min_disp = 1 / max_depth
    depth = 1 / (min_disp + (max_disp - min_disp) * disp)
    return depth


def rotation_matrix(axisangles):
    bsize = axisangles.shape[0]
    R = torch.zeros((bsize, 4, 4), device=axisangles.device)
    theta = torch.norm(axisangles, p=2, dim=1, keepdim=True)
    axis = axisangles / (theta + 1e-10)
    
    a = axis[:, 0:1]
    b = axis[:, 1:2]
    c = axis[:, 2:3]

    cos = torch.cos(theta)
    icos = 1 - cos
    sin = torch.sin(theta)

    R[:,0,0] = cos + a**2*icos
    R[:,0,1] = a*b*icos - c*sin
    R[:,0,2] = a*c*icos + b*sin
    R[:,1,0] = a*b*icos + c*sin
    R[:,1,1] = cos + b**2*icos
    R[:,1,2] = b*c*icos - a*sin
    R[:,2,0] = a*c*icos - b*sin
    R[:,2,1] = b*c*icos + a*sin
    R[:,2,2] = cos + c**2*icos
    R[:,3,3] = 1.0
    return R


def translation_matrix(translation):
    bsize = translation.shape[0]
    T = torch.zeros((bsize, 4, 4), device=translation.shape)

    T[:,0,0] = 1.0
    T[:,1,1] = 1.0
    T[:,2,2] = 1.0
    T[:,3,3] = 1.0
    T[:,:3,3] = translation
    return T


# Source: https://github.com/nianticlabs/monodepth2/blob/ab2a1bf7d45ae53b1ae5a858f4451199f9c213b3/layers.py#L218
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)