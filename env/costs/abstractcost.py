class AbstractCost:
    def __init__(self):
        raise NotImplementedError

    def stage_cost(self, xt, ut, terminal=False):
        """ Quadratic state cost
        :Parameters:
            - xt:           torch.tensor with shape (batch_size, Nx)
            - ut:           torch.tensor with shape (batch_size, Nu)
            - terminal:     bool; whether this is the terminal state
        """
        raise NotImplementedError
