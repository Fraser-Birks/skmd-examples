import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader, Subset



def assemble_dataset(D, ref, device: torch.DeviceObjType = None):
    X_tensor = D.to(device)
    E_tensor = ref.potential_energy(D).unsqueeze(1).to(device)
    F_tensor = ref.forces(D).to(device)
    return TensorDataset(X_tensor, E_tensor, F_tensor)  



class ModelTrainer:
    def __init__(
        self,
        loss_fn=None,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        batch_size=32,
        epochs=1000,
        device="cpu",
        print_every=100,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.print_every = print_every

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.epoch_losses = []
        self.batch_losses = []

    def _compute_loss(self, model, batch):
        batch = [b.to(self.device) for b in batch]
        X, E, F = batch

        if isinstance(self.loss_fn, EnergyForceLoss):
            return self.loss_fn(model, X, E, F)
        elif isinstance(self.loss_fn, EnergyLoss):
            return self.loss_fn(model, X, E)
        elif isinstance(self.loss_fn, ForceLoss):
            return self.loss_fn(model, X, F)
        else:
            raise ValueError("Unsupported loss_fn type.")
        

    def _train_epoch(self, model, dataloader, optimizer):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            loss = self._compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            self.batch_losses.append(loss_val)
            total_loss += loss_val

        return total_loss / max(len(dataloader), 1)


    def train(self, model, dataset):
        
        # reset history
        self.epoch_losses.clear()
        self.batch_losses.clear()
        # move model to device
        model = model.to(self.device)
        # define dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # define optimizer
        optimizer = self.optimizer_cls(
            model.parameters(),
            **self.optimizer_kwargs,
        )

        for epoch in range(self.epochs):
            epoch_loss = self._train_epoch(model, dataloader, optimizer)
            self.epoch_losses.append(epoch_loss)

        return {
            "model": model,
            "losses": self.epoch_losses,
        }



class CVModelTrainer:
    def __init__(
        self,
        k_folds=5,
        loss_fn=None,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        batch_size=32,
        epochs=1000,
        device="cpu",
        shuffle=True,
        seed=1234,
    ):
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.shuffle = shuffle
        self.seed = seed

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Outputs/logs
        self.cv_models = []              # list of k trained models
        self.cv_fold_losses = []         # per-fold train loss histories
        self.base_losses = []            # full-data base model loss history

    def _compute_loss(self, model, batch):
        batch = [b.to(self.device) for b in batch]
        X, E, F = batch

        if isinstance(self.loss_fn, EnergyForceLoss):
            return self.loss_fn(model, X, E, F)
        elif isinstance(self.loss_fn, EnergyLoss):
            return self.loss_fn(model, X, E)
        elif isinstance(self.loss_fn, ForceLoss):
            return self.loss_fn(model, X, F)
        else:
            raise ValueError("Unsupported loss_fn type.")

    def _train_epoch(self, model, dataloader, optimizer):
        model.train()
        total = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = self._compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / max(len(dataloader), 1)

    def _make_folds(self, n_samples):
        if self.k_folds < 2:
            raise ValueError("k_folds must be >= 2")
        if self.k_folds > n_samples:
            raise ValueError("k_folds cannot exceed dataset size")

        g = torch.Generator()
        g.manual_seed(self.seed)

        indices = torch.arange(n_samples)
        if self.shuffle:
            indices = indices[torch.randperm(n_samples, generator=g)]

        fold_sizes = [n_samples // self.k_folds] * self.k_folds
        for i in range(n_samples % self.k_folds):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for fs in fold_sizes:
            folds.append(indices[start:start + fs])
            start += fs
        return folds

    def _train_cv_ensemble(self, base_model, dataset):
        self.cv_models.clear()
        self.cv_fold_losses.clear()

        n = len(dataset)
        folds = self._make_folds(n)

        for fold_idx in range(self.k_folds):
            train_idx = torch.cat([folds[j] for j in range(self.k_folds) if j != fold_idx], dim=0)
            train_subset = Subset(dataset, train_idx.tolist())
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

            # clone model architecture/weights
            m = copy.deepcopy(base_model).to(self.device)
            opt = self.optimizer_cls(m.parameters(), **self.optimizer_kwargs)

            fold_train_losses = []
            for _ in range(self.epochs):
                tr_loss = self._train_epoch(m, train_loader, opt)
                fold_train_losses.append(tr_loss)

            self.cv_models.append(m)
            self.cv_fold_losses.append(fold_train_losses)

    def _train_base_full(self, model, dataset):
        self.base_losses.clear()

        model = model.to(self.device)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        opt = self.optimizer_cls(model.parameters(), **self.optimizer_kwargs)

        for _ in range(self.epochs):
            loss = self._train_epoch(model, loader, opt)
            self.base_losses.append(loss)

        return model

    def train(self, model, dataset):
        # train k-fold ensemble
        self._train_cv_ensemble(model, dataset)

        # train base model on full data
        base_model = self._train_base_full(model, dataset)

        return {
            "base_model": base_model,
            "ensemble_models": self.cv_models,
            "losses": self.base_losses,
            "cv_fold_losses": self.cv_fold_losses,
        }
    


class EnergyForceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # energy weight
        self.beta = beta    # force weight
        self.loss = nn.MSELoss()

    def forward(self, model, X, E_true, F_true):
        # Forces require gradients wrt inputs
        X = X.requires_grad_(True)

        # Predict energies and forces
        E_pred = model.potential_energy(X)
        F_pred = model.forces(X)

        loss_E = self.loss(E_pred, E_true)
        loss_F = self.loss(F_pred, F_true)

        return self.alpha * loss_E + self.beta * loss_F
    


class EnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, model, X, E_true):
        # Predict energies
        E_pred = model.potential_energy(X)
        loss_E = self.loss(E_pred, E_true)

        return loss_E
    


class ForceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, model, X, F_true):
        # Forces require gradients wrt inputs
        X = X.requires_grad_(True)

        # Predict forces
        F_pred = model.forces(X)
        loss_F = self.loss(F_pred, F_true)

        return loss_F