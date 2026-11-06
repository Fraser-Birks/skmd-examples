import torch
from typing import NamedTuple

class RBFKernel:
    def __init__(self,
            alpha, 
            ls,
        ):
        self.alpha = alpha
        self.ls = ls
    
    def compute_kernel(self, x1, x2):
        return self.alpha * torch.exp(-0.5 * torch.norm(x1 - x2)**2 / self.ls**2)
    
    def compute_kernel_batch(self, X, x):
        diff = X - x.unsqueeze(0)
        sqdist = (diff**2).sum(dim=1)
        return self.alpha * torch.exp(-sqdist / (2*self.ls**2))
    
    def compute_grad_kernel(self, x1, x2):
        # gradient with respect to first argument x1
        k = self.compute_kernel(x1, x2)
        return -k * (x1 - x2) / self.ls**2
    
    def compute_grad_kernel_batch(self, X, x):
        diff = X - x.unsqueeze(0)
        K = self.compute_kernel_batch(X, x)
        return -diff / (self.ls**2) * K.unsqueeze(1)
    
    def compute_trH_batch(self, X, x):
        diff = X - x.unsqueeze(0)
        sqdist = (diff ** 2).sum(dim=1)
        K = self.compute_kernel_batch(X, x)

        d = X.shape[1]
        ls2 = self.ls ** 2
        return K * (d / ls2 - sqdist / (ls2 ** 2))


# kernel bandwidth functions
def median_kernel_bandwidth(xset, kernel):
    D = torch.cdist(xset, xset, p=2)
    N = xset.size(0)
    i,j = torch.triu_indices(N, N, offset=1)
    pair_dist = D[i,j]

    ls = torch.sqrt(torch.median(pair_dist))
    kernel.ls = ls

def mean_kernel_bandwidth(xset, kernel):
    D = torch.cdist(xset, xset, p=2)
    N = xset.size(0)
    i,j = torch.triu_indices(N, N, offset=1)
    pair_dist = D[i,j]

    ls = torch.sqrt(torch.mean(pair_dist))
    kernel.ls = ls


def const_kernel_bandwidth(xset, kernel):
    return