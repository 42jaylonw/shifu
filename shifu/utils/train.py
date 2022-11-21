import torch


class HistoryRecorder:
    def __init__(self, shape, num_history, device):
        assert isinstance(num_history, int) and num_history > 0, "num_history must be a positive int"
        self.dshape = shape
        self.num_history = num_history
        self.device = device
        self.history_buf = torch.zeros(*shape, num_history, device=device)

    def add(self, x):
        self.history_buf[..., 1:] = self.history_buf[..., :-1].clone()
        self.history_buf[..., 0] = x

    def reset_idx(self, idx):
        self.history_buf.index_fill_(0, idx, 0.)

    def get_last(self, idx):
        """
        get t-x history value
        Args:
            idx:
                0 means current
                1 means previous t-1 history


        Returns:
            x means previous x history
        """
        return self.history_buf[..., idx]

    def flatten(self):
        p = self.history_buf.permute(0, *reversed(range(1, len(self.history_buf.shape))))
        return p.reshape(*self.dshape[:-1], self.dshape[-1] * self.num_history)


if __name__ == '__main__':
    hs = HistoryRecorder(shape=(10, 3), num_history=3, device='cpu')
    a = torch.Tensor([1, 1, 1]).repeat((10, 1))
    hs.add(a)
    hs.add(a + 1)
    hs.add(a + 2)
    hs.reset_idx(torch.Tensor([0, 1, 2, 7, 8, 9]).to(torch.long))
    z = hs.flatten()
    print()
