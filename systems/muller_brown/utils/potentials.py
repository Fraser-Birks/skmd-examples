import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Type, Tuple, Dict
 


class MullerBrown:
    def __init__(self, dtype=torch.float32, device: torch.DeviceObjType = None):
        self.device = device
        self.A = torch.tensor([-200., -100., -170., 15.], dtype=dtype, device=device)
        self.a = torch.tensor([-1., -1., -6.5, 0.7], dtype=dtype, device=device)
        self.b = torch.tensor([0., 0., 11., 0.6], dtype=dtype, device=device)
        self.c = torch.tensor([-10., -10., -6.5, 0.7], dtype=dtype, device=device)
        self.x0 = torch.tensor([1., 0., -0.5, -1.], dtype=dtype, device=device)
        self.y0 = torch.tensor([0., 0.5, 1.5, 1.], dtype=dtype, device=device)
        self.dtype = dtype

    def potential_energy(self, coord):
        coord = torch.atleast_2d(coord).to(self.device)
        x = coord[:, 0][:, None]
        y = coord[:, 1][:, None]

        dx = x - self.x0
        dy = y - self.y0

        exp_term = self.a * dx**2 + self.b * dx * dy + self.c * dy**2
        V = 0.05 * torch.sum(self.A * torch.exp(exp_term), dim=1)
        return V if len(coord) > 1 else V[0]

    def forces(self, coord):
        coord = torch.atleast_2d(coord).to(self.device)
        x = coord[:, 0][:, None]
        y = coord[:, 1][:, None]

        dx = x - self.x0
        dy = y - self.y0

        exp_term = torch.exp(self.a * dx**2 + self.b * dx * dy + self.c * dy**2)
        coeff = 0.05 * self.A * exp_term

        fx = torch.sum(coeff * (2 * self.a * dx + self.b * dy), dim=1)
        fy = torch.sum(coeff * (self.b * dx + 2 * self.c * dy), dim=1)

        forces = torch.stack([-fx, -fy], dim=1)
        return forces if len(coord) > 1 else forces[0]
    


class MLPPotential(nn.Module):
    def __init__(self,
            hidden_dim=20,
            n_hidden_layers=1,
            activation: Type[nn.Module] = nn.ReLU,
            device: torch.DeviceObjType = None,
        ):
        super(MLPPotential, self).__init__()
        layers = [nn.Linear(2, hidden_dim, device=device), activation()]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, 1, device=device))  # Output: atomic energy
        self.network = nn.Sequential(*layers)
        self.dtype = torch.float32
        self.device = device

    def forward(self, coord):
        coord = coord.to(self.device).clone().detach().requires_grad_(True)

        dx = coord[:, 0] + 1 
        dy = coord[:, 1] - 1
        confine = 0.75 * torch.exp(0.7*dx**2 + 0.6*dx*dy + 0.7*dy**2)
        energy = self.network(coord) + confine.unsqueeze(1) # shape: [batch_size, 1]

        # Compute forces: F = -∇E
        forces = -torch.autograd.grad(
            outputs=energy,
            inputs=coord,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True
        )[0]  # shape: [batch_size, 2]
    
        return forces
    
    def potential_energy(self, coord):
        coord = coord.to(self.device).clone().detach().requires_grad_(True)
        coord = torch.atleast_2d(coord)
        dx = coord[:, 0] + 1
        dy = coord[:, 1] - 1
        confine = 0.75 * torch.exp(0.7*dx**2 + 0.6*dx*dy + 0.7*dy**2)
        e = self.network(coord) + confine.unsqueeze(-1)
        return e if len(coord) > 1 else e[0]

    def forces(self, coord):
        coord = torch.atleast_2d(coord).to(self.device)
        forces = self.forward(coord)
        return forces if len(coord) > 1 else forces[0]
    