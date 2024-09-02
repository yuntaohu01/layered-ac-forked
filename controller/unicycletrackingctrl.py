import torch
from .abstractcontroller import AbstractController


class UnicyclePDController(AbstractController):
    """ The control law is given as
        [ a ] = [cos(theta) -v*sin(theta)]^{-1} (x_ref'' - e_x - 2*e_x')
        [ w ]   [sin(theta) v*cos(theta) ]      (y_ref'' - e_y - 2*e_y')
    """

    def __init__(self, dt=0.1):
        self.Nx = 4
        self.Nu = 2
        self.dt = dt

    def control(self, x0_refs):
        """ Return the tracking control action
        :Parameters:
            - x0_refs:  torch.Tensor(batch_size, Nx * T). We assume the input
                        comes in the form of (x0, r0, r1, r2, ...).flatten()
        """
        batch_size = x0_refs.size(0)
        u = torch.zeros((batch_size, self.Nu), device=x0_refs.device)

        for i, x0_ref in enumerate(x0_refs):
            x0_ref = x0_ref.reshape(-1, self.Nx)
            pos_ref, vel_ref, acc_ref = construct_full_ref(
                    x0_ref[1:3], dt=self.dt
            )
            xx, yy, v, theta = x0_ref[0]
            Ainv = torch.tensor(
                    [[v * torch.cos(theta), v * torch.sin(theta)],
                     [-torch.sin(theta), torch.cos(theta)]],
                    device=x0_refs.device
            ) / v
            pos = x0_ref[0][:2]
            vel = torch.tensor([
                v * torch.cos(theta), v * torch.sin(theta)],
                device=x0_refs.device
            )
            u[i] = Ainv @ (acc_ref[0] + (pos_ref[0] - pos) +
                           (vel_ref[0] - vel) * 2)

        return u


def construct_full_ref(x_ref, dt, u_ref=None):
    """ Convert a reference trajectory of the 4D state into (pos, vel, acc)
    :Parameters:
        - x_ref:    torch.Tensor(T, 4) (unbatched reference state traj)
        - u_ref:    torch.Tensor(T, 2) (reference control input)
    :Returns:
        - p_ref:    torch.Tensor(T, 2) (2 entrys for x and y directions)
        - v_ref:    torch.Tensor(T, 2)
        - a_ref:    torch.Tensor(T, 2)
    """
    p_ref = x_ref[:, :2]
    v_ref = torch.vstack([
        x_ref[:, 2] * torch.cos(x_ref[:, 3]),
        x_ref[:, 2] * torch.sin(x_ref[:, 3]),
    ]).T
    if u_ref is None:
        pedal = (x_ref[1:, 2] - x_ref[:-1, 2]) / dt
        steer = (x_ref[1:, 3] - x_ref[:-1, 3]) / dt
        a_ref = torch.vstack([
            pedal * torch.cos(x_ref[:-1, 3]) - x_ref[:-1, 2] * steer * torch.sin(x_ref[:-1, 3]),
            pedal * torch.sin(x_ref[:-1, 3]) + x_ref[:-1, 2] * steer * torch.cos(x_ref[:-1, 3]),
        ]).T
    else:
        a_x = u_ref[:, 0] * torch.cos(x_ref[:-1, 3]) - \
                u_ref[:, 1] * x_ref[:-1, 2] * torch.sin(x_ref[:-1, 3]) 
        a_y = u_ref[:, 0] * torch.sin(x_ref[:-1, 3]) + \
                u_ref[:, 1] * x_ref[:-1, 2] * torch.cos(x_ref[:-1, 3])
        a_ref = torch.vstack([a_x, a_y]).T
    return p_ref, v_ref, a_ref
