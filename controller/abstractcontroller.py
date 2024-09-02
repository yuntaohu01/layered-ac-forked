import torch


class AbstractController:
    """ Abstract class of controllers, to be inherited by the other classes
    """

    def __init__(self):
        self.counter = 0

    def reset(self):
        self.counter = 0

    def control(self, x):
        raise NotImplementedError
