import torch
import numpy as np
from utils.potentials import *
from scipy.spatial.distance import pdist
import warnings

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# root mean squared error (RMSE)
def compute_RMSE(testset, model):
    model.eval()
    Xtrue, Etrue, Ftrue = testset[:]
    Epred = model.potential_energy(Xtrue)
    Fpred = model.forces(Xtrue)

    rmse_e = torch.sqrt(torch.sum(Etrue - Epred)**2 / len(Etrue))
    rmse_f = torch.sqrt(torch.sum(torch.norm(Ftrue - Fpred,dim=1)**2) / len(Ftrue))

    return rmse_e.detach().item(), rmse_f.detach().item()


def RMSE_over_timesteps(al_logger, sim_logger, testset, n_hidden, save_dir):
    train_steps = al_logger["train_steps"]
    tot_steps = len(sim_logger["traj"])+1
    rmse_e = np.zeros(tot_steps)
    rmse_f = np.zeros(tot_steps)

    for n in range(len(train_steps)):
        train_step = train_steps[n]
        # load nth model iterate
        model_n = MLPPotential(hidden_dim=n_hidden, n_hidden_layers=1, activation=nn.Tanh, device=torch.device("cpu"))
        model_n.load_state_dict(torch.load(save_dir + f"model_{train_step}.pt"))
        # compute RMSE in energies and forces
        rmse_e_n, rmse_f_n = compute_RMSE(testset, model_n)

        if n < len(train_steps)-1:
            train_step_next = train_steps[n+1]
            rmse_e[train_step:train_step_next] = np.ones(train_step_next - train_step) * rmse_e_n
            rmse_f[train_step:train_step_next] = np.ones(train_step_next - train_step) * rmse_f_n
        else:
            rmse_e[train_step:] = np.ones(tot_steps - train_step) * rmse_e_n
            rmse_f[train_step:] = np.ones(tot_steps - train_step) * rmse_f_n

    return rmse_e, rmse_f


# kernel Stein discrepancy (KSD) U-statistic
def compute_KSD(X, scores, kernel_band):
    N, d = X.shape
    U = _compute_KSD_U_matrix(X, scores, kernel_band)
    # remove diagonal for U-statistic
    mask = ~torch.eye(N, dtype=torch.bool)

    ksd2 = U[mask].sum() / (N * (N - 1))

    return ksd2


def _compute_KSD_U_matrix(X, scores, kernel_band):
    """
    Build the pairwise Stein kernel matrix U_ij used in KSD U-statistics.
    """
    N, d = X.shape
    h2 = kernel_band**2

    diff = X[:, None, :] - X[None, :, :]      # (N,N,d)
    sqdist = (diff ** 2).sum(-1)              # (N,N)

    K = torch.exp(-sqdist / (2 * h2))         # (N,N)
    score_dot = scores @ scores.T             # (N,N)

    grad_x_k = -(diff / h2) * K[..., None]    # (N,N,d)
    grad_y_k = (diff / h2) * K[..., None]     # (N,N,d)

    term1 = score_dot * K
    term2 = (scores[:, None, :] * grad_y_k).sum(-1)
    term3 = (scores[None, :, :] * grad_x_k).sum(-1)
    term4 = K * (d / h2 - sqdist / (h2 ** 2))

    U = term1 + term2 + term3 + term4
    return U


def ksd_goodness_of_fit_test(
    X,
    scores,
    kernel_band,
    M=500,
    alpha=0.05,
    scale_by_n=False,
):
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for U-statistic.")

    # observed statistic from existing function
    U_n = compute_KSD(X, scores, kernel_band)  # KSD^2 U-stat
    T_obs = (n * U_n) if scale_by_n else U_n

    # build U_ij matrix once for bootstrap
    U = _compute_KSD_U_matrix(X, scores, kernel_band)
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)

    probs = torch.full((n,), 1.0 / n, device=X.device)
    boot = torch.empty(M, device=X.device, dtype=torch.float32)

    for m in range(M):
        # multinomial counts c_i
        idx = torch.multinomial(probs, num_samples=n, replacement=True)
        c = torch.bincount(idx, minlength=n).to(dtype=X.dtype, device=X.device)

        # normalized + centered weights
        w = c / n
        wc = w - (1.0 / n)

        W = wc[:, None] * wc[None, :]          # (n,n)
        Tb = (W[mask] * U[mask]).sum()
        if scale_by_n:
            Tb = n * Tb
        boot[m] = Tb

    # p-value from bootstrap tail
    p_value = (torch.sum(boot >= T_obs).item() + 1.0) / (M + 1.0)
    boot_np = boot.detach().cpu().numpy()
    crit = float(np.quantile(boot_np, 1.0 - alpha))
    reject = bool(T_obs.item() > crit)

    return {
        "ksd2_obs": float(U_n.item()),
        "test_stat_obs": float(T_obs.item()),
        "bootstrap_test_stats": boot_np,
        "p_value": float(p_value),
        "reject": reject,
        "critical_value": crit,
        "alpha": float(alpha),
        "M": int(M),
        "n": int(n),
        "scale_by_n": bool(scale_by_n),
    }


def ksd_goodness_of_fit_over_sets(
    samples,
    score_fn,
    kernel_band,
    Nsets=10,
    Nsamp=10_000,
    M=100,
    alpha=0.05,
    scale_by_n=False,
):
    T = len(samples)

    pvals = np.zeros(Nsets, dtype=float)
    rejects = np.zeros(Nsets, dtype=bool)
    ksd2_obs = np.zeros(Nsets, dtype=float)
    t_obs = np.zeros(Nsets, dtype=float)

    for i in range(Nsets):

        ids = torch.randperm(T)[:Nsamp]
        samp_set = samples[ids]
        score_set = score_fn(samp_set)

        res = ksd_goodness_of_fit_test(
            X=samp_set,
            scores=score_set,
            kernel_band=kernel_band,
            M=M,
            alpha=alpha,
        )
        pvals[i] = res["p_value"]
        rejects[i] = res["reject"]
        ksd2_obs[i] = res["ksd2_obs"]
        t_obs[i] = res["test_stat_obs"]

    out = {
        "p_values": pvals,
        "rejects": rejects,
        "ksd2_obs": ksd2_obs,
        "test_stat_obs": t_obs,
        "alpha": float(alpha),
        "M": int(M),
        "N": int(Nsets),
        "scale_by_n": bool(scale_by_n),
    }

    return out


def _autocorr_fft(x):
    x = np.asarray(x)
    n = len(x)

    x = x - np.mean(x)
    fft = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(fft * np.conjugate(fft))[:n]

    acf /= acf[0]  # normalize

    return acf


def compute_ESS(chain):
    chain = np.asarray(chain)

    if chain.ndim == 1:
        chain = chain[:, None]

    N, D = chain.shape
    ess = np.zeros(D)

    for d in range(D):
        x = chain[:, d]

        acf = _autocorr_fft(x)

        tau = 1.0
        for k in range(1, len(acf)):
            if acf[k] + acf[k+1] < 0:
                break
            tau += 2 * acf[k]

        ess[d] = N / tau

    return ess if D > 1 else ess[0]



def compute_pairwise_diversity_metrics(X: np.ndarray, eps: float = 1e-10) -> Dict[str, float]:
    """
    mpd: mean pairwise distance
    med_pd: median pairwise distance
    logdet_cov: log determinant of the covariance
    """
    n, d = X.shape

    # pairwise distances
    D = pdist(X, metric="euclidean")
    mpd = float(np.mean(D))
    med_pd = float(np.median(D))

    # covariance spread
    C = np.cov(X.T)
    if np.ndim(C) == 0:  # edge fallback
        C = np.array([[float(C)]])
    C = C + eps * np.eye(C.shape[0])
    sign, logdet = np.linalg.slogdet(C)
    logdet_cov = float(logdet) if sign > 0 else -np.inf

    return {"mpd": mpd, "med_pd": med_pd, "logdet_cov": logdet_cov}


def compute_W2_sinkhorn(
    X,
    Y,
    reg=1e-2,
    normalize=True,
    method="sinkhorn",      # "stabilized" or "sinkhorn"
    numItermax=1000,
    stopThr=1e-8,
):
    import ot  # POT (pip install pot), only needed for Wasserstein distances
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if normalize:
        Z = np.vstack([X, Y])
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
        Y = (Y - mu) / sd

    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)

    C = ot.dist(X, Y, metric="sqeuclidean").astype(np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Warning: numerical errors at iteration")
        if method == "stabilized":
            w2_sq = ot.bregman.sinkhorn2(
                a, b, C, reg,
                method="sinkhorn_stabilized",
                numItermax=numItermax,
                stopThr=stopThr,
                warn=False
            )
        else:
            w2_sq = ot.sinkhorn2(
                a, b, C, reg,
                warn=False
            )

    # fallback if NaN/Inf
    if not np.isfinite(w2_sq):
        w2_sq = ot.bregman.sinkhorn2(
            a, b, C, reg=max(reg, 1e-1),
            method="sinkhorn_stabilized",
            numItermax=10000,
            stopThr=1e-7,
            warn=False
        )

    return float(np.sqrt(max(w2_sq, 0.0)))


def balanced_mc_sinkhorn_w2(
    X, Y_ref, w2_fn, repeats=50, seed=0
):
    """
    X: (n,d) current training subset
    Y_ref: (N,d) large reference pool
    w2_fn: callable(Xsub, Ysub) -> scalar W2 (e.g., robust_w2_sinkhorn)
    """
    X = np.asarray(X)
    Y_ref = np.asarray(Y_ref)
    n = X.shape[0]
    N = Y_ref.shape[0]
    if n > N:
        raise ValueError(f"n={n} > N_ref={N}")

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        idx = rng.choice(N, size=n, replace=False)
        Y = Y_ref[idx]
        vals.append(w2_fn(X, Y))
    vals = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "all": vals
    }


def compute_W2_sinkhorn_over_trials(
    X,
    Y,
    reg=1e-2,
    normalize=True,
    seed=None,
):
    """Progressive-subsampling Sinkhorn W2: trial i samples len(Y) points from the first len(Y)*(i+1) of X.

    Returns an array of floor(len(X)/len(Y)) W2 values.
    """
    import ot  # POT (pip install pot), only needed for Wasserstein distances

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D tensors")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have same dimensionality")

    N = X.shape[0]
    M = Y.shape[0]

    if M > N:
        raise ValueError("Y cannot be larger than X")

    device = X.device

    if seed is not None:
        torch.manual_seed(seed)

    if normalize:
        Z = torch.cat([X, Y], dim=0)
        mean = Z.mean(dim=0)
        std = Z.std(dim=0)
        std[std == 0] = 1.0

        X = (X - mean) / std
        Y = (Y - mean) / std

    # number of trials
    n_trials = N // M

    # weights for OT (numpy, required by POT)
    a = np.ones(M) / M
    b = np.ones(M) / M

    w2_vals = []

    for i in range(n_trials):
        pool_size = M * (i + 1)
        X_pool = X[:pool_size]

        if i == 0:
            # deterministic: first M elements
            X_sub = X_pool
        else:
            idx = torch.randperm(pool_size, device=device)[:M]
            X_sub = X_pool[idx]

        # cost matrix (squared Euclidean)
        C = torch.cdist(X_sub, Y, p=2) ** 2
        C_np = C.detach().cpu().numpy()

        W2_sq = ot.sinkhorn2(a, b, C_np, reg)
        w2_vals.append(np.sqrt(W2_sq))

    return np.array(w2_vals)