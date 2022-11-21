import torch
from torch import nn


class Module(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device

    def loss_func(self, pred, label):
        raise NotImplementedError

    def save(self, logdir):
        torch.save(self.state_dict(), f"{logdir}/{self.__class__.__name__}.pt")

    def load(self, logdir):
        self.load_state_dict(torch.load(f"{logdir}/{self.__class__.__name__}.pt",
                                        map_location=self.device))
        self.eval()
