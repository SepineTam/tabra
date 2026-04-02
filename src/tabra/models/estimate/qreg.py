#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : qreg.py

import numpy as np
from scipy import optimize, stats as sp_stats

from tabra.models.estimate.base import BaseModel
from tabra.results.qreg_result import QRegResult


class QuantileRegression(BaseModel):
    def fit(self, df, y, x, quantile=0.5, is_con=True):
        """Fit a quantile regression model.

        Args:
            df (pd.DataFrame): Input dataset.
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            quantile (float): Target quantile (0, 1). Default 0.5 (median).
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            QRegResult: Quantile regression result.

        Example:
            >>> dta = load_data("auto")
            >>> result = QuantileRegression().fit(dta._df, "price", ["weight", "mpg"], quantile=0.25)
        """
        df = self._prepare_df(df, y, x)
        tau = quantile
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)

        if is_con:
            X = np.column_stack([X, np.ones(X.shape[0])])
            var_names = var_names + ["_cons"]

        n, k = X.shape

        # Solve quantile regression via linear programming
        beta = self._solve_lp(X, y_vec, tau)

        # Fitted values and residuals
        fitted = X @ beta
        resid = y_vec - fitted

        # Sum of absolute deviations (minimized)
        sum_adev = float(np.sum(self._rho(resid, tau)))

        # Raw sum of deviations: sum rho_tau(y_i - q_y)
        # where q_y is the tau-quantile of y
        q_y = self._stata_quantile(y_vec, tau)
        sum_rdev = float(np.sum(self._rho(y_vec - q_y, tau)))

        # Pseudo R-squared
        pseudo_r2 = 1.0 - sum_adev / sum_rdev if sum_rdev > 0 else 0.0

        # Degrees of freedom
        df_model = k - 1 if is_con else k
        df_resid = n - k

        # VCE: iid standard errors using kernel density estimation
        vce, f_r, sparsity, bwidth = self._compute_vce_iid(
            X, y_vec, beta, resid, fitted, tau, n, k
        )

        std_err = np.sqrt(np.diag(vce))
        t_stat = beta / std_err
        p_value = np.array([2 * (1 - sp_stats.t.cdf(abs(t), df_resid)) for t in t_stat])

        return QRegResult(
            coef=beta,
            std_err=std_err,
            t_stat=t_stat,
            p_value=p_value,
            vce=vce,
            n_obs=n,
            k_vars=k,
            df_model=df_model,
            df_resid=df_resid,
            var_names=var_names,
            y_name=y,
            quantile=tau,
            q_v=q_y,
            pseudo_r2=pseudo_r2,
            sum_adev=sum_adev,
            sum_rdev=sum_rdev,
            f_r=f_r,
            sparsity=sparsity,
            bwidth=bwidth,
            resid=resid,
            fitted=fitted,
        )

    def _solve_lp(self, X, y, tau):
        """Solve quantile regression via linear programming.

        Decision variables: z = [beta(k), u(n), v(n)]
        where u = max(resid, 0), v = max(-resid, 0), resid = y - X*beta.

        Objective: minimize tau * 1'u + (1-tau) * 1'v
        Constraints: X*beta + u - v = y, u >= 0, v >= 0
        """
        n, k = X.shape

        # Objective: c = [0_k, tau*1_n, (1-tau)*1_n]
        c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])

        # Equality: [X | I | -I] z = y
        A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
        b_eq = y

        # Bounds: beta unbounded, u >= 0, v >= 0
        bounds = [(None, None)] * k + [(0, None)] * (2 * n)

        result = optimize.linprog(
            c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method='highs',
            options={'maxiter': 10000},
        )

        if not result.success:
            beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
            return self._solve_irls(X, y, tau, beta_init)

        beta = result.x[:k]
        return beta

    def _solve_irls(self, X, y, tau, beta_init, max_iter=100, tol=1e-8):
        """Fallback: Iteratively reweighted least squares for quantile regression."""
        beta = beta_init.copy()
        n = len(y)

        for _ in range(max_iter):
            resid = y - X @ beta
            # Weights for IRLS: 1 / (|resid| + epsilon)
            weights = 1.0 / (np.abs(resid) + 1e-6)
            # Asymmetric weights for quantile
            w = np.where(resid >= 0, tau * weights, (1 - tau) * weights)
            W = np.diag(w)

            try:
                beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        return beta

    def _rho(self, u, tau):
        """Check function rho_tau(u) = u * (tau - I(u < 0))."""
        return u * (tau - (u < 0).astype(float))

    def _stata_quantile(self, y, tau):
        """Compute quantile matching Stata's convention for qreg.

        Stata uses floor(tau * n) as the index (0-based) into sorted y.
        """
        y_sorted = np.sort(y)
        n = len(y)
        idx = int(np.floor(tau * n))
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        return float(y_sorted[idx])

    def _compute_vce_iid(self, X, y, beta, resid, fitted, tau, n, k):
        """Compute VCE under iid assumption using 'fitted' density method.

        Uses Hall-Sheather bandwidth and the empirical quantile function
        of fitted values to estimate the sparsity (inverse density).
        """
        # Hall-Sheather bandwidth
        z_alpha = sp_stats.norm.ppf(0.975)
        z_tau = sp_stats.norm.ppf(tau)
        phi_z = sp_stats.norm.pdf(z_tau)

        h = n**(-1/3) * z_alpha**(2/3) * (
            1.5 * phi_z**2 / (2 * phi_z**2 + tau * (1 - tau))
        )**(1/3)

        # Fitted method: sparsity via quantile function of fitted values
        # s(tau) = [Q(tau+h) - Q(tau-h)] / (2 * h * phi(invnorm(tau)))
        qlo = max(0.0, tau - h)
        qhi = min(1.0, tau + h)
        delta = np.percentile(fitted, qhi * 100) - np.percentile(fitted, qlo * 100)
        sparsity = delta / (2 * h * phi_z) if h * phi_z > 0 else 0.0

        # Density at the quantile
        f_r = 1.0 / sparsity if sparsity > 0 else 0.0

        # VCE: tau*(1-tau) / f_r^2 * (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)
        if f_r > 0:
            vce = (tau * (1 - tau) / (f_r**2)) * XtX_inv
        else:
            vce = tau * (1 - tau) * XtX_inv

        return vce, f_r, sparsity, h

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for quantile regression estimation")
