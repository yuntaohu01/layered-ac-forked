import numpy as np

def sim_forward(self, controller, T, x0s=None):
    if x0s is None:
        x0s = self.random_x0()
    batch_size = x0s.size(0)
    x = torch.zeros((batch_size, T+1, self.Nx),
                    dtype=torch.double, device=self.device)
    u = torch.zeros((batch_size, T, self.Nu),
                    dtype=torch.double, device=self.device)
    cost = 0
    x[:, 0] = x0s
    for t in range(T):
        ut = controller.control(x[:, t])
        x[:, t+1], c = self.step(x[:, t], ut)
        u[:, t] = ut
        cost = cost + c
    return x, u, cost

def random_x0(self, num_samples=1, seed=None):
    """ Generate a random initial condition """
    if seed is not None:
        torch.manual_seed(seed)
    toret = torch.randn((num_samples, self.Nx),
                        dtype=torch.double, device=self.device)
    return toret
