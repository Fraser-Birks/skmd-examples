import torch
from utils.errormetrics import *

class TimeInterval:
    def __init__(self,
            interval
        ):
        self.interval = interval
    
    def evaluate(self, outdict):
        step = outdict["step"]
        if step is None:
            raise ValueError("TimeInterval requires 'step'")
        return (step % self.interval)
    
    def is_activated(self, outdict):
        return self.evaluate(outdict) == 0
    
    def update(self):
        return
    

class MaxKernelEval:
    def __init__(self,
            buffer,
            thresh
        ):
        self.counter = 0
        self.buffer = buffer
        self.thresh = thresh
    
    def evaluate(self, outdict):
        kernels = outdict["K"]
        if kernels is None:
            raise ValueError("MaxKernelEval requires 'K'")  
        return torch.max(kernels)
    
    def is_activated(self, outdict):
        self.counter += 1
        return (self.counter >= self.buffer and self.evaluate(outdict) < self.thresh)
    
    def update(self):
        self.counter = 0
        return
    


class SteinForce:
    def __init__(self,
            buffer,
            thresh
        ):
        self.counter = 0
        self.buffer = buffer
        self.thresh = thresh
    
    def evaluate(self, outdict):
        force = outdict["Fskmd"]
        if force is None:
            raise ValueError("SteinForce requires 'Fskmd'")  
        return torch.norm(force)
    
    def is_activated(self, outdict):
        self.counter += 1
        return (self.counter >= self.buffer and self.evaluate(outdict) < self.thresh)
    
    def update(self):
        self.counter = 0
        return



class SteinPoints:
    """
    Evaluate the empirical average:
      alpha(x) = (1/J) * sum_i kappa_pi(x_i, x)

    and activate when alpha(x) < threshold after buffer steps.

    Using Stein kernel cross-term (x_i, x):
      kappa_pi(x_i, x)
        = s_i^T s_x * k(x_i, x)
          + s_i^T grad_x k(x_i, x)
          + s_x^T grad_{x_i} k(x_i, x)
          + tr(grad_{x_i} grad_x k(x_i, x))
    """
    def __init__(self,
            buffer,
            thresh
        ):
        self.counter = 0
        self.buffer = buffer
        self.thresh = thresh

    def evaluate(self, outdict):
        x = outdict["x"]
        F = outdict["F"]
        Fsys = outdict["Fsys"]
        K = outdict["K"]
        gradK = outdict["gradK"]
        traceH = outdict["traceH"]


        if K is None or gradK is None or F is None or Fsys is None:
            raise ValueError("SteinPoints requires 'K', 'gradK', 'F', and 'Fsys'")
        
        if F.ndim == 2:
            F = F.squeeze(0)
        if x.ndim == 2:
            x = x.squeeze(0)

        # term1: s_i^T s_x * k(x_i, x)
        term1 = (Fsys * F.unsqueeze(0)).sum(dim=1) * K                        
        # Need grad_x k(x_i, x). For RBF, grad_x k = -grad_{x_i} k
        gradK_x = -gradK                            
        # term2: s_i^T grad_x k(x_i, x)
        term2 = (F.unsqueeze(0) * gradK_x).sum(dim=1)
        # term3: s_x^T grad_{x_i} k(x_i, x)
        term3 = (Fsys * gradK).sum(dim=1)

        # term4: trace Hessian cross
        if traceH is None:
            term4 = 0
        else:
            term4 = traceH                                    

        kappa = term1 + term2 + term3 + term4
        return kappa.mean()

    def is_activated(self, outdict):
        self.counter += 1
        return (self.counter >= self.buffer and self.evaluate(outdict) < self.thresh)

    def update(self):
        self.counter = 0
        return
    


class KernelSteinDiscr:
    def __init__(self,
            buffer,
            thresh
        ):
        self.counter = 0
        self.buffer = buffer
        self.thresh = thresh
    
    def evaluate(self, outdict):
        positions = outdict["positions"]
        forces = outdict["Ffix"]
        kernel_band = outdict["kernel_band"]
        
        return compute_KSD(positions, forces, kernel_band)
    
    def is_activated(self, outdict):
        self.counter += 1
        return (self.counter >= self.buffer and self.evaluate(outdict) < self.thresh)
    
    def update(self):
        self.counter = 0
        if self.thresh > -5:
            self.thresh -= 0.25
        return

    
    
class EnsembleUncertainty:
    def __init__(self,
            buffer,
            thresh
        ):
        self.counter = 0
        self.buffer = buffer
        self.thresh = thresh
    
    def evaluate(self, outdict):
        rho = outdict["rho"]
        if rho is None:
            raise ValueError("EnsembleUncertainty requires 'rho'")
        return rho
    
    def is_activated(self, outdict):
        self.counter += 1
        return (self.counter >= self.buffer and self.evaluate(outdict) > self.thresh)
    
    def update(self):
        self.counter = 0
        return