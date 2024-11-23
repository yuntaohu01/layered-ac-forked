import torch

from .abstractdynamics import AbstractDynamics


class LinearDynamics(AbstractDynamics):
    def __init__(self, A, B):
        """ Initialize the environment. """
        self.A = A.to(A.device)
        self.B = B.to(A.device)
        self.Nx, self.Nu = B.shape
        self.device = self.A.device

    def step(self, x, u):
        """ Step the dynamics forward
        :Parameters:
            - x: torch.Tensor of shape (batch, Nx)
            - u: torch.Tensor of shape (batch, Nu)
        """
        x_next = self.A @ x.T + self.B @ u.T
        return x_next.T


def sample_linear_dynamics(Nx, Nu, A_norm=0.995, B_norm=1, device='cpu', seed = None):
    """Generates dynamics matrices (A,B) on a specified device.

    Parameters:
        - Nx: int, number of state variables
        - Nu: int, number of input variables
        - A_norm: float, operator norm of matrix A (default: 0.995)
        - B_norm: float, operator norm of matrix B (default: 1)
        - device: str, the device to use ('cpu' or 'cuda')

    Returns:
        - lin_dym: LinearDynamics object with dynamics (A, B) on the specified device
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    A = torch.randn(Nx, Nx, device=device)
    B = torch.randn(Nx, Nu, device=device)

    # Normalize A and B
    A = A / torch.linalg.norm(A, ord=2) * A_norm
    B = B / torch.linalg.norm(B, ord=2) * B_norm
    return LinearDynamics(A, B)

def integrator(order, decay=1.):
    """ Generate a n-th order intergrator where
    A = [[1, d, 0, 0, 0, ...]
         [0, 1, d, 0, 0, ...]
         [0, 0, 1, d, 0, ...]
         ... ]
    B = [[0], [0], ..., [1]]
    Parameters:
        - order:    int, the order of the integrator
        - decay:    float, d in the matrix A above
    Returns:
        - dyn:      LinearDynamics object with dynamics (A, B)
    """
    A = torch.eye(order + 1) + torch.eye(order + 2)[1:, :-1] * decay
    B = torch.zeros((order + 1, 1))
    B[-1] = 1
    return LinearDynamics(A, B)
