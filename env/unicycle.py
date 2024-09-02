import torch


class Unicycle(AbstractEnv):
    """ Discrete time unicycle system with the 4D state and 2D input
        x = [x, y, velocity, heading angle]
        u = [acceleration, angular velocity] """

    def __init__(self, device=None, dt=1e-2):
        self.Nx = 4
        self.Nu = 2
        self.dt = dt
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def step(self, x, u):
        dx = self.dt * torch.hstack([
            x[:, 2] * torch.cos(x[:, 3]),
            x[:, 2] * torch.sin(x[:, 3]),
            u[:, 0],
            u[:, 1]])
        # TODO: implement cost; now just set all state cost to 0
        return x + dx, 0

    def traj_cost(self, x, u):
        return (x * x).sum(1) + (u * u).sum(1)

    def random_x0(self, num_samples, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        toret = torch.randn((num_samples, self.Nx),
                            dtype=torch.float, device=self.device)
        toret[:, 2] = 1e-2
        toret[:, 3] /= 3.14
        return toret


def tracking_normalization(x, ref):
    def alpha_to_r(alpha):
        return torch.tensor([[torch.cos(-alpha), -torch.sin(-alpha)],
                             [torch.sin(-alpha), torch.cos(-alpha)]])
    assert x.size(0) == ref.size(0)
    batch_size, Nx = x.shape
    ref = ref.reshape(batch_size, -1, Nx)
    Rs = torch.stack(
        [alpha_to_r(x[i, 3]).T for i in range(batch_size)]
    ).double()

    norm_ref = ref.clone()
    norm_ref[:, :, :2] = torch.bmm(norm_ref[:, :, :2] - x[:, None, :2], Rs)
    norm_ref[:, :, 3] = norm_ref[:, :, 3] - x[:, None, 3]

    xx = torch.zeros_like(x)
    xx[:, 2] = x[:, 2]
    return xx, norm_ref
