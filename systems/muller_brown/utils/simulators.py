import torch
from typing import NamedTuple
from utils.training import *
from utils.kernels import *
from utils.triggers import *



def simulate_trajectory(simulator, potential, x0=None):
    # check that devices are consistent
    if potential.device != simulator.device:
        raise ValueError("Device type of potential and simulator do not match.")

    # optionally initialize trajectory
    if "traj" in simulator.logger and x0 is not None:
        x0_t = torch.as_tensor(x0, device=simulator.device, dtype=simulator.logger["traj"].dtype)

        traj0_shape = simulator.logger["traj"][0].shape
        x0_shape = torch.atleast_2d(x0_t).shape

        # shape match
        if x0_t.shape == traj0_shape:
            simulator.logger["traj"][0] = x0_t
        elif x0_shape == traj0_shape:
            simulator.logger["traj"][0] = torch.atleast_2d(x0_t)
        else:
            raise ValueError(f"x0 shape {tuple(x0_t.shape)} incompatible with simulator traj[0] shape {tuple(traj0_shape)}")

    torch.manual_seed(simulator.seed)

    for t in range(simulator.T - 1):
        simulator.sim_step(t, potential)

    return



def simulate_ensemble(simulator, potential, x0, lag=10, trigger=None, trigger_key="trigger"):
    """
    Simulate with optional trigger evaluation at every step.

    If trigger is provided:
      - evaluate trigger on sim_step output
      - store value in simulator.logger[trigger_key] as a tensor of length T-1
    """
    if potential.device != simulator.device:
        raise ValueError("Device type of potential and simulator do not match.")

    # initiate logger fields
    simulator.logger["traj"][0] = torch.atleast_2d(x0)
    torch.manual_seed(simulator.seed)

    if trigger is not None:
        simulator.logger[trigger_key] = torch.zeros(simulator.T - 1, device=simulator.device)

    for t in range(simulator.T - 1):
        output = simulator.sim_step(t, potential)

        # evaluate & log trigger if provided
        if trigger is not None:
            trig_val = trigger.evaluate(output)
            # ensure scalar tensor
            if not torch.is_tensor(trig_val):
                trig_val = torch.tensor(trig_val, device=simulator.device, dtype=simulator.logger["traj"].dtype)
            simulator.logger[trigger_key][t] = trig_val

        # switch particle
        if t % lag == 0:
            simulator.update_sim_params(t)

    return


class ALRoutine:
    def __init__(self,
            simulator,
            ref,
            model,
            x0,
            trigger,
            trainer,
            batch_size,
            initdata: torch.Tensor,
            max_iter = 100,
        ):
        self.simulator = simulator
        self.ref = ref
        self.model = model
        self.x0 = x0
        self.trigger = trigger
        self.trainer = trainer
        self.batch_size = batch_size
        self.initdata = initdata
        self.max_iter = max_iter
        if model.device != simulator.device:
            raise ValueError("Device type of potential and simulator do not match.")
        else:
            self.device = model.device

        # initiate logger fields
        self.logger = {
            "trigger": torch.zeros(simulator.T-1, device=self.device),
            "train_steps": [0],
            "train_data": initdata,
            "train_loss": [],
            "batch_size": batch_size,
        }
        self.simulator.logger["traj"][0] = torch.atleast_2d(x0)
        torch.manual_seed(simulator.seed)


    # active learning with greedy (per particle) criterion
    def active_learn(self, savedir, T = None):
        # set number of simulation steps
        if T is None:
            T = self.simulator.T
        torch.save(self.model.state_dict(), savedir + f"model_0.pt")

        datact = 0
        trainct = 0
        for t in range(0, T-1):
            # take one simulation step
            output = self.simulator.sim_step(t, self.model)
            self.logger["trigger"][t] = self.trigger.evaluate(output)
            
            if self.trigger.is_activated(output):
                # add current point to training data
                newdata = self.simulator.logger["traj"][t].detach().unsqueeze(0)
                self.logger["train_data"] = torch.cat([self.logger["train_data"], newdata], dim=0)
                datact += 1
                # update simulation parameters
                self.simulator.update_sim_params(t) # e.g. switch to other point
                self.trigger.update()

                if datact == self.batch_size:
                    trainct += 1
                    # retrain model
                    self.logger["train_steps"].append(t)
                    dataset = assemble_dataset(self.logger["train_data"], self.ref, device=self.device)
                    train_res = self.trainer.train(self.model, dataset)
                    losses = train_res["losses"]
                    self.logger["train_loss"] += losses
                    torch.save(self.model.state_dict(), savedir + f"model_{t}.pt")
                    # update ensemble if UDD
                    if hasattr(self.simulator, "ensemble"):
                        self.simulator.ensemble = train_res["ensemble_models"]
                    # reset parameters
                    datact = 0
                    print(f"Training at iteration {t}: loss {losses[-1]}")

            # once max number of iterations reached, break out of loop
            if trainct >= self.max_iter:
                # truncate logged data
                self.logger["trigger"] = self.logger["trigger"][:t+1]
                self.simulator.logger["timesteps"] = self.simulator.logger["timesteps"][:t+1]
                self.simulator.logger["traj"] = self.simulator.logger["traj"][:t+1]
                break

        # save logger data
        torch.save(self.logger, savedir + "al_logger.pt")
        torch.save(self.simulator.logger, savedir + "sim_logger.pt")
        return
    

    # active learning with ensemble criterion
    def active_learn_ens(self, savedir, lag = 100, T = None):
        ti = TimeInterval(lag)
        # set number of simulation steps
        if T is None:
            T = self.simulator.T
        torch.save(self.model.state_dict(), savedir + f"model_0.pt")

        for t in range(0, T-1):
            # take one simulation step
            output = self.simulator.sim_step(t, self.model)
            self.logger["trigger"][t] = self.trigger.evaluate(output)
            
            if ti.is_activated(output):
                self.simulator.update_sim_params(t)

            if self.trigger.is_activated(output):
                self.trigger.update()
                # add data
                newdata = self.simulator.logger["sys"][t].detach()
                self.logger["train_data"] = torch.cat([self.logger["train_data"], newdata], dim=0)
                # retrain
                self.logger["train_steps"].append(t)
                dataset = assemble_dataset(self.logger["train_data"], self.ref, device=self.device)
                losses = self.trainer.train(self.model, dataset)
                self.logger["train_loss"] += losses
                torch.save(self.model.state_dict(), savedir + f"model_{t}.pt")
                print(f"Training at iteration {t}: loss {losses[-1]}")

        torch.save(self.logger, savedir + "al_logger.pt")
        torch.save(self.simulator.logger, savedir + "sim_logger.pt")
        return
    



class OverdampedLangevin:
    def __init__(self,
            beta, 
            dt,
            T,
            device: torch.DeviceObjType = None,
            seed: int = 1234,
        ):
        self.beta = beta
        self.dt = dt
        self.T = T
        self.diff_coeff = torch.sqrt(torch.tensor(2 * dt / beta, device=device))
        self.device = device
        self.seed = seed
        self.logger = {
            "timesteps": torch.arange(0, T*dt, dt, device=device),
            "traj": torch.empty((T, 2), device=device),
            "seed": seed,
        }

    def sim_step(self, i, potential):
        noise = torch.randn(2, device=self.device)
        coord = self.logger["traj"][i]
        self.logger["traj"][i+1] = coord + potential.forces(coord) * self.dt + self.diff_coeff * noise
        output = {
            "step": i
        }
        return output
    
    def update_sim_params(self, i):
         return
    

class EnsembleLangevin:
    def __init__(self,
            sys,                   
            beta,
            dt,
            T,
            device: torch.DeviceObjType = None,
            seed: int = 1234,
        ):
        self.sys = sys.to(device)
        self.beta = beta
        self.dt = dt
        self.T = T
        self.device = device
        self.seed = seed

        self.diff_coeff = torch.sqrt(torch.tensor(2 * dt / beta, device=device))

        J, d = self.sys.shape
        self.J = J
        self.d = d

        self.logger = {
            "timesteps": torch.arange(0, T * dt, dt, device=device),
            "traj": torch.empty((T, J, d), device=device),
            "seed": seed,
        }
        # initialize first time slice with starting coordinates
        self.logger["traj"][0] = self.sys.clone()

    def sim_step(self, i, potential):
        coord = self.logger["traj"][i]
        noise = torch.randn((self.J, self.d), device=self.device)
        F = potential.forces(coord) 

        # Euler-Maruyama update
        self.logger["traj"][i + 1] = coord + F * self.dt + self.diff_coeff * noise

        output = {
            "step": i,
        }
        return output

    def update_sim_params(self, i):
        return
    

class SKMD:
    def __init__(self,
            kernel,
            set_kernel_band, # kernel bandwidth function
            sys,
            sys_fix,
            A,
            beta, 
            dt,
            T,
            device: torch.DeviceObjType = None,
            seed: int = 1234,
        ):
        self.kernel = kernel
        self.set_kernel_band = set_kernel_band
        self.sys = sys
        self.sys_fix = sys_fix
        self.A = A
        self.beta = beta
        self.dt = dt
        self.T = T
        self.diff_coeff = torch.sqrt(torch.tensor(2 * dt * 31 / len(sys_fix), device=device)) # set \eta=len(sys_fix)
        self.device = device
        self.seed = seed
        self.logger = {
            "timesteps": torch.arange(0, T*dt, dt, device=device),
            "sys": torch.empty((T, len(sys)+1, 2), device=device),
            "traj": torch.empty((T, 2), device=device),
            "seed": seed,
        }
     

    def compute_kernelized_forces(self, x):
        # set kernel bandwidth
        augset = torch.cat([self.sys, x.unsqueeze(0)], dim=0)
        self.set_kernel_band(augset, self.kernel)
        # compute kernel and gradient kernel
        K = self.kernel.compute_kernel_batch(self.sys, x)
        gradK = self.kernel.compute_grad_kernel_batch(self.sys, x)
        traceH = self.kernel.compute_trH_batch(self.sys, x)
        return K, gradK, traceH


    def sim_step(self, i, potential):
        noise = torch.randn(2, device=self.device)
        coord = self.logger["traj"][i]

        # standard Langevin force
        F = self.beta * potential.forces(coord)
        # SKMD force
        Fsys = self.beta * potential.forces(self.sys)
        K, gradK, traceH = self.compute_kernelized_forces(coord)
        Fskmd = (Fsys * K.unsqueeze(1) + gradK).mean(dim=0)

        self.logger["traj"][i+1] = coord + (self.A * F + Fskmd) * self.dt + self.diff_coeff * noise
        self.logger["sys"][i] = torch.cat([self.sys, coord.unsqueeze(0)], dim=0)

        output = {
            "step": i,
            "K": K,               # k(x_i, x)
            "gradK": gradK,       # grad_{x_i} k(x_i, x)
            "traceH": traceH,     # tr(grad_{x_i} grad_x k(x_i, x))
            "F": F,               # grad log pi(x_i)
            "Fsys": Fsys,         # grad log pi(x)
            "Fskmd": Fskmd,
            "positions": torch.cat([self.sys, coord.unsqueeze(0)], dim=0),
            "x": coord,           # current point
        }
        return output
    

    def update_sim_params(self, i):
        # assign first element of sys_fix as the next starting point of the simulation
        self.logger["traj"][i+1] = self.sys[0]
        # add current point to the end of the active set
        newpt = self.logger["traj"][i].detach().unsqueeze(0)
        self.sys = torch.cat([self.sys[1:], newpt], dim=0)
        return 
    

class UDD:
    def __init__(self,
            ensemble, # list/tuple of M model potentials
            mag,
            ls,
            beta,
            dt,
            T,
            Natom = 1, # number of atoms for rho normalization
            device: torch.DeviceObjType = None,
            seed: int = 1234,
        ):
        self.ensemble = ensemble
        self.mag = mag
        self.ls = ls
        self.dt = dt
        self.T = T
        self.Natom = Natom
        self.M = len(ensemble)
        self.diff_coeff = torch.sqrt(torch.tensor(2 * dt / beta, device=device))
        self.device = device
        self.seed = seed
        self.logger = {
            "rho": torch.zeros(T-1, device=self.device),
            "timesteps": torch.arange(0, T * dt, dt, device=device),
            "traj": torch.empty((T, 2), device=device),
            "seed": seed,
        }


    def _ensemble_energies_forces(self, x):
        energies = []
        forces = []
        for m in self.ensemble:
            e = torch.atleast_1d(m.potential_energy(x)).squeeze()   
            f = m.forces(x)
            energies.append(e)
            forces.append(f)
        E = torch.stack(energies)
        F = torch.stack(forces, dim=0)
        return E, F


    def _udd_force(self, x):
        E, F = self._ensemble_energies_forces(x)
        Ebar = E.mean()
        Fbar = F.mean(dim=0)

        dE = E - Ebar
        dF = F - Fbar

        sigma2 = 0.5 * torch.sum(dE ** 2)

        C = self.Natom * self.M * (self.ls ** 2)
        exp_term = torch.exp(-sigma2 / C)
        dEbias_dsigma2 = self.mag * exp_term * (-1.0 / C)

        neg_grad_sigma2 = torch.sum(dE.unsqueeze(1) * dF, dim=0)

        F_bias = dEbias_dsigma2 * neg_grad_sigma2
        V_udd = self.mag * (exp_term - 1.0)

        # uncertainty metric
        rho = torch.sqrt(torch.tensor(2.0 / (self.M * self.Natom), device=x.device)) * torch.sqrt(sigma2)

        return F_bias, V_udd, sigma2, rho


    def sim_step(self, i, potential):
        noise = torch.randn(2, device=self.device)
        coord = self.logger["traj"][i]

        # standard Langevin force from supplied potential
        F_base = potential.forces(coord)

        # UDD bias force (analytic)
        F_bias, V_udd, sigma2, rho = self._udd_force(coord)

        self.logger["traj"][i+1] = coord + (F_base + F_bias) * self.dt + self.diff_coeff * noise
        self.logger["rho"][i] = rho

        output = {
            "step": i,
            "F_bias": F_bias,
            "V_udd": V_udd,
            "sigma2": sigma2,
            "rho": rho,
        }
        return output

    def update_sim_params(self, i):
        return
    

class StochasticSVGD:
    def __init__(self,
            kernel,
            set_kernel_band,
            sys,
            beta,
            dt,
            T,
            device: torch.DeviceObjType = None,
            seed: int = 1234,
            jitter: float = 1e-8,
        ):
        self.kernel = kernel
        self.set_kernel_band = set_kernel_band
        self.sys = sys.to(device)
        self.beta = beta
        self.dt = dt
        self.T = T
        self.device = device
        self.seed = seed
        self.jitter = jitter
        self.diff_coeff = torch.sqrt(torch.tensor(2 * dt / beta, device=device))

        J, d = self.sys.shape
        self.J = J
        self.d = d

        self.logger = {
            "timesteps": torch.arange(0, T * dt, dt, device=device),
            "sys": torch.empty((T, J, d), device=device),
            "seed": seed,
        }
        self.logger["sys"][0] = self.sys.clone()

    def _compute_kernel_mats(self, X):
        J, d = X.shape
        K = torch.empty((J, J), device=self.device)
        gradK = torch.empty((J, J, d), device=self.device)

        for j in range(J):
            xj = X[j]
            K[:, j] = self.kernel.compute_kernel_batch(X, xj)
            gradK[:, j, :] = self.kernel.compute_grad_kernel_batch(X, xj)

        return K, gradK

    def _chol_factor(self, K):
        """
        Compute lower-triangular L such that K ≈ L L^T using Cholesky.
        Adds adaptive jitter if needed.
        """
        Ks = 0.5 * (K + K.T)  # symmetrize numerically
        J = Ks.shape[0]
        I = torch.eye(J, device=Ks.device, dtype=Ks.dtype)

        jitter = self.jitter
        max_tries = 8
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(Ks + jitter * I)
                return L, Ks, jitter
            except RuntimeError:
                jitter *= 10.0

        raise RuntimeError(
            f"Cholesky failed even after jitter escalation; final jitter={jitter:.2e}"
        )

    def sim_step(self, i, potential):
        X = self.sys
        self.set_kernel_band(X, self.kernel)

        
        F = potential.forces(X)

        # Kernel and kernel gradients
        K, gradK = self._compute_kernel_mats(X)

        # SVGD velocity field
        attractive = torch.matmul(K.T, F) / self.J
        repulsive = gradK.sum(dim=0) / self.J
        phi = attractive + repulsive

        # Stochastic update
        noise = torch.randn(2, device=self.device)

        # Euler-Maruyama update
        X_next = X + (phi * self.dt) + (self.diff_coeff * noise)

        self.sys = X_next
        self.logger["sys"][i + 1] = X_next

        output = {
            "step": i,
            "phi": phi,
            "K": K,
            "kernel_band": self.kernel.ls if hasattr(self.kernel, "ls") else None,
        }
        return output

    def update_sim_params(self, i):
        return