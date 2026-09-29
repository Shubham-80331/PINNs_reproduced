import torch
import numpy as np
from scipy.stats import qmc

def latin_hypercube_sample(N, bounds):
    """
    Generate N Latin Hypercube samples within the given bounds.
    bounds: list of [lower, upper] for each dimension.
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d)
    sample = sampler.random(n=N)
    
    # Scale samples
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    scaled_sample = lower_bounds + sample * (upper_bounds - lower_bounds)
    return scaled_sample

def numpy_to_tensor(x, requires_grad=False):
    """Convert numpy array to PyTorch tensor."""
    tensor = torch.tensor(x, dtype=torch.float32)
    if requires_grad:
        tensor.requires_grad = True
    return tensor
