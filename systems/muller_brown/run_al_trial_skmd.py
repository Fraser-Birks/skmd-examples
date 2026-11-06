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
from utils.training import assemble_dataset, EnergyForceLoss, ModelTrainer
from utils.simulators import ALRoutine, SKMD, simulate_trajectory
from utils.triggers import TimeInterval, SteinForce, SteinPoints
from utils.kernels import RBFKernel, const_kernel_bandwidth, mean_kernel_bandwidth
from utils.plotting import plot_trigger, plot_al_iteration
from utils.errormetrics import RMSE_over_timesteps

# Shared inputs (initial model, initial data, test set); see data/MANIFEST.md.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "muller_brown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--outdir", type=str, default="results/timeint_trigger/")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)

    # simulation / AL params
    parser.add_argument("--T", type=int, default=160000)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)  # overwritten below from init data
    parser.add_argument("--max-iter", type=int, default=16)

    # SKMD kernel params
    parser.add_argument("--k-alpha", type=float, default=2.0)
    parser.add_argument("--k-ls", type=float, default=0.41)

    # trigger params: "time" = SKMD (fixed interval), "stein-force" = a-SKMD (Eq. 8)
    parser.add_argument("--trigger", choices=["time", "stein-force"], default="time")
    parser.add_argument("--time-interval", type=int, default=250)
    parser.add_argument("--buffer", type=int, default=100)
    parser.add_argument("--sp-thresh", type=float, default=1.0) # SteinPoints: 12.0

    # model/training
    parser.add_argument("--hidden-dim", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)

    prefix = "askmd" if args.trigger == "stein-force" else "skmd"
    trial_dir = os.path.join(args.outdir, f"{prefix}_A={args.A}_run{args.trial_id}/")
    os.makedirs(trial_dir, exist_ok=True)

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
        raise FileNotFoundError(f"Expected test dataset at {args.data_dir}/dataset_test_1000.torch")

    dataset = assemble_dataset(D0, ref, device=device)
    testset = assemble_dataset(Dtest, ref, device=device)

    # Keep one point as initial x0; remainder as fixed/ensemble set
    start_ind = -3
    x0 = D0[start_ind]
    mask = torch.ones(D0.shape[0], dtype=torch.bool, device=device)
    mask[start_ind] = False
    sys0 = D0[mask]             # active set for SKMD
    sys_fix = sys0.clone()      # optional compatibility

    fixdatalen = len(sys0)      # query this many points before retraining

    # --- trainer ---
    loss_fn = EnergyForceLoss(alpha=0.5, beta=1.0)
    trainer = ModelTrainer(
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )

    # --- kernel + simulator: SKMD ---
    kernel = RBFKernel(alpha=args.k_alpha, ls=args.k_ls)

    sim = SKMD(
        kernel=kernel,
        set_kernel_band=const_kernel_bandwidth,
        sys=sys0,
        sys_fix=sys_fix,
        A = args.A,
        beta=args.beta,
        dt=args.dt,
        T=args.T,
        device=device,
        seed=trial_seed,
    )

    # initialize trajectory via simulate_trajectory utility
    simulate_trajectory(sim, model0, x0=x0)

    # --- trigger ---
    if args.trigger == "stein-force":
        trigger = SteinForce(buffer=args.buffer, thresh=args.sp_thresh)
    else:
        trigger = TimeInterval(args.time_interval)


    # --- AL routine ---
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

    # compute test error
    rmse_e, rmse_f = RMSE_over_timesteps(al.logger, al.simulator.logger, testset, args.hidden_dim, trial_dir)
    np.savez(trial_dir + "test_error.npz", rmse_e, rmse_f)

    # iteration plots
    limits = [[-2., 1.5], [-0.5, 2.5]]
    lvls = np.arange(-11, 30, 1)
    for train_iter in range(1, len(al.logger["train_steps"])):  # starts at 1 because plotting uses train_iter-1
        plot_al_iteration(
            train_iter,
            al.logger,
            al.simulator.logger,
            args.hidden_dim,
            D0,
            fixdatalen,
            lvls,
            limits,
            trial_dir,
            titlelabel="SKMD",
            show_legend=False,
            save_fig=True
        )

    print(f"[DONE] Trial {args.trial_id} saved to {trial_dir}")

if __name__ == "__main__":
    main()