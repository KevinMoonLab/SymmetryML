import numpy as np
from typing import Any, Dict, List, Tuple
from . import register_projection


def _svd_solve_penalty(J, rhs, mu, svd_rel=1e-3, svd_abs=1e-12, tikhonov=0.0):
    n = J.shape[1] if J.ndim == 2 else 0
    if J.size == 0:
        return rhs / (1.0 + tikhonov), {'rank': 0, 'sigma_max': 0.0, 'tau': svd_abs}
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_max = S[0] if S.size > 0 else 0.0
    tau = max(svd_rel * sigma_max, svd_abs)
    rhs_V = Vt @ rhs
    diag = 1.0 + mu * (S**2) + tikhonov
    mask = S >= tau
    denom = np.where(mask, diag, 1.0 + tikhonov)
    Delta_V = rhs_V.copy()
    k = S.size
    Delta_V[:k] = Delta_V[:k] / denom
    if n > k:
        Delta_V[k:] = Delta_V[k:] / (1.0 + tikhonov)
    Delta = Vt.T @ Delta_V
    return Delta, {'rank': int(mask.sum()), 'sigma_max': float(sigma_max), 'tau': float(tau)}


def _penalty_objective(y, x, mu, g_fun):
    g = g_fun(y)
    return 0.5 * np.dot(y - x, y - x) + 0.5 * mu * np.dot(g, g)


def _armijo(phi, y, Delta, f0, c=1e-4, max_backtracks=12):
    alpha = 1.0
    for _ in range(max_backtracks):
        if phi(y + alpha * Delta) <= f0 - c * alpha * np.dot(Delta, Delta):
            return alpha
        alpha *= 0.5
    return alpha


@register_projection('penalty-homotopy')
def project_penalty_homotopy(
    lse,
    X: np.ndarray,
    mu_min: float = 1e-6,
    mu_max: float = 1e6,
    mu_growth: float = 100.0,
    target_drop: float = 10.0,
    window: int = 5,
    min_progress_ratio: float = 0.7,
    max_stages: int = 8,
    max_it_per_stage: int = 20,
    tol_res: float = 1e-8,
    tol_step: float = 1e-8,
    svd_rel: float = 1e-3,
    svd_abs: float = 1e-12,
    tikhonov: float = 0.0,
    line_c: float = 1e-4,
    max_backtracks: int = 12,
    # --- New knobs ---
    use_data_mean_init: bool = False,        # optional: start from dataset mean if it helps
    tiny_mu_for_init: float = 1e-12,        # tiny μ used only to compare initial objective
    target_step_init: float = None,         # if None, computed from ||x||
    return_info: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any] | List[Dict[str, Any]]]:

    X = np.asarray(X, dtype=np.float64)
    single = False
    if X.ndim == 1:
        X = X[None, :]
        single = True

    def g_fun(y_vec):
        return lse.constraint_values(y_vec[None, :])[0]

    def J_fun(y_vec):
        return lse.get_constraint_jacobian(y_vec[None, :])[0]

    # Try to obtain ambient data mean if needed
    def _get_X_mean():
        if getattr(lse, "X_mean_", None) is not None:
            return np.asarray(lse.X_mean_, dtype=np.float64)
        X_train = getattr(lse, "X_", None)
        if X_train is not None:
            return np.mean(np.asarray(X_train, dtype=np.float64), axis=0)
        return None

    Y = X.copy()
    infos: List[Dict[str, Any]] = []

    for i in range(X.shape[0]):
        x = X[i]
        y = x.copy()

        # Diagnostics: residual at the query x (independent of chosen start)
        g_at_x = g_fun(x)
        r_at_x = float(np.linalg.norm(g_at_x))

        # --- (Patch 2) Optional: choose dataset mean as starting iterate y0 if it helps ---
        started_from = 'x'
        if use_data_mean_init:
            x_mean = _get_X_mean()
            if x_mean is not None and x_mean.shape == x.shape:
                g_x = g_fun(y)         # at y = x (current)
                g_try = g_fun(x_mean)  # at candidate y = mean
                r_x = float(np.linalg.norm(g_x))
                r_try = float(np.linalg.norm(g_try))

                # Compare tiny-μ objectives to avoid bias from constraint term
                phi_x = 0.5 * np.dot(y - x, y - x) + 0.5 * tiny_mu_for_init * np.dot(g_x, g_x)
                phi_try = 0.5 * np.dot(x_mean - x, x_mean - x) + 0.5 * tiny_mu_for_init * np.dot(g_try, g_try)

                if (r_try < r_x) and (phi_try <= phi_x):
                    y = x_mean.copy()
                    started_from = 'x_mean'

        # Residual at chosen start y0
        g0 = g_fun(y)
        r0 = float(np.linalg.norm(g0))
        if r0 <= tol_res:
            Y[i] = y
            infos.append({
                'stages': 0, 'iterations': 0,
                'initial_residual': r0,
                'initial_residual_at_x': r_at_x,
                'final_residual': r0,
                'distance_moved': 0.0,
                'avg_rank_kept': 0.0,
                'sigma_max_seen': 0.0,
                'backtracks': 0,
                'mu_final': mu_min,
                'mu_init': mu_min,
                'started_from': started_from,
                'status': 'already_on_manifold'
            })
            continue

        # --- (Patch 1) Adaptive μ initialization based on ||J^T g|| scale ---
        J0 = J_fun(y)
        JTg0 = J0.T @ g0
        s0 = float(np.linalg.norm(JTg0))
        if s0 < 1e-16:
            # fallback: s0 ≈ sigma_max(J) * ||g||
            if J0.size > 0:
                U0, S0, Vt0 = np.linalg.svd(J0, full_matrices=False)
                sigma_max0 = S0[0] if S0.size > 0 else 1.0
            else:
                sigma_max0 = 1.0
            s0 = sigma_max0 * max(r0, 1e-16)

        if target_step_init is None:
            target_step = min(0.5, max(1e-3, float(np.linalg.norm(x))))
        else:
            target_step = float(target_step_init)

        mu_init = target_step / (s0 + 1e-12)
        mu = float(np.clip(mu_init, mu_min, mu_max))

        stage = 0
        it_total = 0
        backtracks_total = 0
        best = {'y': y.copy(), 'r': r0}
        target_this_stage = r0 / target_drop
        recent = [r0]
        avg_rank = 0.0
        rank_count = 0
        sigma_max_seen = 0.0
        status = 'ok'

        phi = lambda z: _penalty_objective(z, x, mu, g_fun)

        while stage < max_stages and mu <= mu_max:
            for _ in range(max_it_per_stage):
                it_total += 1
                g = g_fun(y)
                r = float(np.linalg.norm(g))
                if r < best['r']:
                    best = {'y': y.copy(), 'r': r}
                if r <= tol_res:
                    status = 'tol_res'
                    break

                J = J_fun(y)
                rhs = -(y - x) - mu * (J.T @ g)

                Delta, diag = _svd_solve_penalty(J, rhs, mu, svd_rel=svd_rel, svd_abs=svd_abs, tikhonov=tikhonov)
                avg_rank += diag['rank']; rank_count += 1
                sigma_max_seen = max(sigma_max_seen, diag['sigma_max'])

                # --- (Patch 3) Guard against premature tol_step when residual is still large ---
                if np.linalg.norm(Delta) <= tol_step:
                    r_now = r  # current residual
                    if r_now <= max(10 * tol_res, 0.1 * r0):
                        status = 'tol_step'
                        break
                    else:
                        # too small step but still far → escalate mu and continue
                        mu = min(mu * mu_growth, mu_max)
                        phi = lambda z, mu_cur=mu: _penalty_objective(z, x, mu_cur, g_fun)
                        continue

                f0 = _penalty_objective(y, x, mu, g_fun)
                alpha = _armijo(phi, y, Delta, f0, c=line_c, max_backtracks=max_backtracks)
                if alpha < 1.0:
                    backtracks_total += 1

                y = y + alpha * Delta
                recent.append(float(np.linalg.norm(g_fun(y))))
                if len(recent) > window:
                    recent.pop(0)

                # Stage advancement rule (residual-informed)
                if r <= target_this_stage:
                    stage += 1
                    mu = min(mu * mu_growth, mu_max)
                    target_this_stage = max(r / target_drop, tol_res)
                    recent = [r]
                    phi = lambda z, mu_cur=mu: _penalty_objective(z, x, mu_cur, g_fun)
                    break

                # If progress stalls over the recent window, gently increase mu
                if len(recent) == window and recent[-1] / recent[0] > min_progress_ratio:
                    mu = min(mu * (mu_growth ** 0.5), mu_max)  # micro-adjust
                    phi = lambda z, mu_cur=mu: _penalty_objective(z, x, mu_cur, g_fun)

            if status in ('tol_res', 'tol_step'):
                break

            # If we exhausted inner iterations without meeting target, escalate mu to next stage
            if r > target_this_stage and mu < mu_max:
                stage += 1
                mu = min(mu * mu_growth, mu_max)
                target_this_stage = max(r / target_drop, tol_res)
                recent = [r]
                phi = lambda z, mu_cur=mu: _penalty_objective(z, x, mu_cur, g_fun)

        y_final = y if float(np.linalg.norm(g_fun(y))) <= best['r'] + 1e-16 else best['y']
        Y[i] = y_final
        final_res = float(np.linalg.norm(g_fun(y_final)))

        infos.append({
            'stages': stage,
            'iterations': it_total,
            'initial_residual': r0,
            'initial_residual_at_x': r_at_x,
            'final_residual': final_res,
            'distance_moved': float(np.linalg.norm(y_final - x)),
            'avg_rank_kept': (avg_rank / max(rank_count, 1)),
            'sigma_max_seen': sigma_max_seen,
            'backtracks': backtracks_total,
            'mu_final': mu,
            'mu_init': float(mu_init),
            'started_from': started_from,
            'status': status if final_res <= tol_res or status != 'ok' else 'max_stages_or_mu'
        })

    if single:
        return Y[0], infos[0] if return_info else Y[0]
    return Y, infos if return_info else Y
