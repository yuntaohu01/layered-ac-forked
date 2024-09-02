import torch

def alpha_to_r(alpha):
    return torch.tensor([[torch.cos(-alpha), -torch.sin(-alpha)],
                         [torch.sin(-alpha), torch.cos(-alpha)]],
                        device = alpha.device)

def batch_alpha_to_R(alphas):
    cos_alphas = torch.cos(alphas)
    sin_alphas = torch.sin(alphas)
    Rs = torch.stack(
            (cos_alphas, -sin_alphas, sin_alphas, cos_alphas),
            dim=1
         ).view(-1, 2, 2)
    return Rs

def unicycle_tracking_normalization(x_ref):
    x, ref = x_ref[:, :4], x_ref[:, 4:]
    batch_size, Nx = x.shape
    ref = ref.reshape(batch_size, -1, Nx)

    Rs = batch_alpha_to_R(x[:, 3])
    norm_ref = ref.clone()
    norm_ref[:, :, :2] = torch.bmm(norm_ref[:, :, :2] - x[:, None, :2], Rs)
    norm_ref[:, :, 3] = norm_ref[:, :, 3] - x[:, None, 3]

    xx = torch.zeros_like(x)
    xx[:, 2] = x[:, 2]
    return torch.hstack([xx, norm_ref.flatten(1)])
