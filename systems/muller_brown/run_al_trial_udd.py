#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import resource

from utils.potentials import MullerBrown, MLPPotential
from utils.training import assemble_dataset, EnergyForceLoss, CVModelTrainer
from utils.simulators import ALRoutine, UDD
from utils.triggers import EnsembleUncertainty
from utils.plotting import plot_trigger, plot_al_iteration
from utils.errormetrics import RMSE_over_timesteps

# Shared inputs (initial model, initial data, test set); see data/MANIFEST.md.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "muller_brown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--outdir", type=str, default="results/udd_trigger/")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)

    # simulation / AL params
    parser.add_argument("--T", type=int, default=160000)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=16)

    # UDD params
    parser.add_argument("--mag", type=float, default=1.0)
    parser.add_argument("--ls", type=float, default=0.2)
    parser.add_argument("--N", type=int, default=1)  # set appropriately for your system

    # trigger params
    parser.add_argument("--buffer", type=int, default=100)
    parser.add_argument("--rho-thresh", type=float, default=0.01)

    # model/training
    parser.add_argument("--hidden-dim", type=int, default=20)
    parser.add_argument("--k-folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)

    # trial-specific output dir
    trial_dir = os.path.join(args.outdir, f"udd_run{args.trial_id}_ls={args.ls}/")
    os.makedirs(trial_dir, exist_ok=True)

    # deterministic seed per trial
    trial_seed = args.seed * args.trial_id
    torch.manual_seed(trial_seed)

    # --- reference system ---
    ref = MullerBrown(device=device)

    # --- initial model ---
    model0 = MLPPotential(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=1,
        activation=nn.Tanh,
        device=device
    )

    # optional warm start
    if os.path.exists(os.path.join(args.data_dir, "model0.torch")):
        model0.load_state_dict(torch.load(os.path.join(args.data_dir, "model0.torch"), map_location=device))

    # --- initial data ---
    if os.path.exists(os.path.join(args.data_dir, "dataset0.torch")):
        D0 = torch.load(os.path.join(args.data_dir, "dataset0.torch"), map_location=device)
    else:
        raise FileNotFoundError(f"Expected initial dataset at {args.data_dir}/dataset0.torch")

    # --- test data ---
    if os.path.exists(os.path.join(args.data_dir, "dataset_test_1000.torch")):
        Dtest = torch.load(os.path.join(args.data_dir, "dataset_test_1000.torch"), map_location=device)
    else:
        raise FileNotFoundError(f"Expected initial dataset at {args.data_dir}/dataset_test_1000.torch")
    
    # initial dataset object
    dataset = assemble_dataset(D0, ref, device=device)
    testset = assemble_dataset(Dtest, ref, device=device)
    fixdatalen = len(D0) - 1

    # --- trainer (CV ensemble + base model) ---
    loss_fn = EnergyForceLoss(alpha=0.5, beta=1.0)
    trainer = CVModelTrainer(
        k_folds=args.k_folds,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        seed=trial_seed,
    )

    trainer._train_cv_ensemble(copy.deepcopy(model0), dataset)
    ensemble_models = trainer.cv_models

    # --- simulator: UDD ---
    sim = UDD(
        ensemble = ensemble_models,
        mag = args.mag,
        ls = args.ls,
        beta=args.beta,
        dt=args.dt,
        T=args.T,
        device=device,
        seed=trial_seed)
    

    # --- trigger ---
    trigger = EnsembleUncertainty(
        buffer=args.buffer,
        thresh=args.rho_thresh
    )

    # --- AL routine ---
    x0 = torch.tensor([-0.55, 1.45], device=device)
    model = copy.deepcopy(model0)
    al = ALRoutine(
        simulator=sim,
        ref=ref,
        model=model,
        x0=x0,
        trigger=trigger,
        trainer=trainer,
        batch_size=fixdatalen,
        initdata=D0,
        max_iter=args.max_iter
    )

    t0 = time.perf_counter()
    al.active_learn(savedir=trial_dir, T=args.T)
    t1 = time.perf_counter()

    # record memory usage
    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss = ru.ru_maxrss
    if os.uname().sysname == "Darwin":
        maxrss_mb = maxrss / (1024**2)   # bytes -> MB
    else:
        maxrss_mb = maxrss / 1024.0      # KB -> MB (Linux)

    runmetrics = {
        "runtime_sec": float(t1 - t0),
        "maxrss_mb": float(maxrss_mb),
    }
    print(f"Runtime: {float(t1 - t0)} sec.")
    print(f"Memory: {float(maxrss_mb)} MB")
    np.savez(trial_dir + "run_metrics.npz", metrics=runmetrics)

    # plot trigger
    fig = plot_trigger(al.logger, trigger, "Ensemble Uncertainty", r"$\rho$")
    fig.savefig(trial_dir + f"plot_trigger.png")

    # compute test error 
    rmse_e, rmse_f = RMSE_over_timesteps(al.logger, al.simulator.logger, testset, args.hidden_dim, trial_dir)
    np.savez(trial_dir + f"test_error.npz", rmse_e, rmse_f)

    # make plots
    limits = [[-2.,1.5],[-0.5,2.5]]
    lvls = np.arange(-11, 30, 1)
    for train_iter in range(len(al.logger["train_steps"])):
        plot_al_iteration(train_iter,
            al.logger,
            al.simulator.logger,
            args.hidden_dim,
            D0,
            fixdatalen,
            lvls,
            limits,
            trial_dir,
            titlelabel="UDD",
            show_legend=False,
            save_fig=True)

    print(f"[DONE] Trial {args.trial_id} saved to {trial_dir}")

if __name__ == "__main__":
    main()