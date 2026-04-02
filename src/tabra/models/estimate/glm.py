#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : glm.py

import numpy as np
from scipy import stats as sp_stats

from tabra.models.estimate.base import BaseModel
from tabra.results.glm_result import GLMResult

# Default link for each family
_DEFAULT_LINK = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "log",
}


class GLMModel(BaseModel):
    """Generalized Linear Model estimated via IRLS."""

    def fit(self, df, y, x, family="gaussian", link=None,
            is_con=True, max_iter=100, tol=1e-8):
        """Fit a GLM via Iteratively Reweighted Least Squares (IRLS).

        Args:
            df (pd.DataFrame): Input dataset.
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            family (str): Distribution family. One of "gaussian", "binomial",
                "poisson", "gamma". Default "gaussian".
            link (str): Link function. Default None (use canonical link for family).
            is_con (bool): Whether to include a constant term. Default True.
            max_iter (int): Maximum IRLS iterations. Default 100.
            tol (float): Convergence tolerance. Default 1e-8.

        Returns:
            GLMResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = GLMModel().fit(dta._df, "price", ["weight", "mpg"], family="gaussian")
        """
        if link is None:
            link = _DEFAULT_LINK.get(family, "identity")

        family = family.lower()
        link = link.lower()

        df = self._prepare_df(df, y, x)
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)

        if is_con:
            X = np.column_stack([X, np.ones(X.shape[0])])
            var_names = var_names + ["_cons"]

        n, k = X.shape

        # Initialize beta via initial mu -> eta -> WLS
        mu_init = self._init_mu(y_vec, family)
        eta_init = self._link_fn(mu_init, link)
        d_eta_d_mu_init = self._d_eta_d_mu(mu_init, link)
        V_mu_init = self._variance_fn(mu_init, family)
        W_init = 1.0 / (V_mu_init * d_eta_d_mu_init ** 2)
        z_init = eta_init + (y_vec - mu_init) * d_eta_d_mu_init
        sqrt_W_init = np.sqrt(np.clip(W_init, 1e-15, 1e15))
        try:
            beta = np.linalg.lstsq(
                X * sqrt_W_init[:, None], z_init * sqrt_W_init, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)

        converged = False
        n_iter = 0

        for iteration in range(max_iter):
            n_iter = iteration + 1
            eta = X @ beta
            mu = self._inv_link_fn(eta, link)

            # Clamp mu for numerical safety
            mu = self._clamp_mu(mu, family)

            # Working response and weights
            d_eta_d_mu = self._d_eta_d_mu(mu, link)
            V_mu = self._variance_fn(mu, family)

            z = eta + (y_vec - mu) * d_eta_d_mu
            W_diag = 1.0 / (V_mu * d_eta_d_mu ** 2)
            W_diag = np.clip(W_diag, 1e-15, 1e15)

            # Weighted least squares: beta = (X'WX)^{-1} X'Wz
            sqrt_W = np.sqrt(W_diag)
            X_w = X * sqrt_W[:, np.newaxis]
            z_w = z * sqrt_W

            try:
                beta_new, _, _, _ = np.linalg.lstsq(X_w, z_w, rcond=None)
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                converged = True
                break

            beta = beta_new

        # Final quantities
        eta = X @ beta
        mu = self._inv_link_fn(eta, link)
        mu = self._clamp_mu(mu, family)

        # VCE from final IRLS iteration
        d_eta_d_mu = self._d_eta_d_mu(mu, link)
        V_mu = self._variance_fn(mu, family)
        W_diag = 1.0 / (V_mu * d_eta_d_mu ** 2)
        W_diag = np.clip(W_diag, 1e-15, 1e15)
        sqrt_W = np.sqrt(W_diag)
        X_w = X * sqrt_W[:, np.newaxis]
        XtWX = X_w.T @ X_w

        try:
            V = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            V = np.linalg.pinv(XtWX)

        std_err = np.sqrt(np.abs(np.diag(V)))
        z_stat = beta / std_err
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # Log-likelihood
        ll = self._log_likelihood(y_vec, mu, family)

        # Null model: fit intercept-only via IRLS
        ll_0 = self._fit_null_model(y_vec, family, link, n)

        # Pseudo R2: McFadden for binomial/poisson, deviance-based for others
        if ll_0 != 0:
            pseudo_r2 = max(1 - ll / ll_0, 0.0)
        else:
            pseudo_r2 = 0.0

        # LR chi2
        df_m = k - 1 if is_con else k
        chi2 = 2 * (ll - ll_0)
        chi2 = max(chi2, 0.0)
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        # Deviance
        deviance = self._deviance(y_vec, mu, family)
        null_deviance = self._deviance(y_vec, np.full(n, np.mean(y_vec)), family)

        model_name = f"GLM ({family}, {link})"

        return GLMResult(
            coef=beta,
            std_err=std_err,
            z_stat=z_stat,
            p_value=p_value,
            ll=ll,
            ll_0=ll_0,
            pseudo_r2=pseudo_r2,
            chi2=chi2,
            chi2_pval=chi2_pval,
            n_obs=n,
            k_vars=k,
            df_m=df_m,
            var_names=var_names,
            y_name=y,
            model_name=model_name,
            converged=converged,
            n_iter=n_iter,
            family=family,
            link=link,
            deviance=deviance,
            null_deviance=null_deviance,
            V=V,
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for GLM estimation")

    @staticmethod
    def _init_mu(y, family):
        """Compute initial mu values for IRLS."""
        if family == "binomial":
            p = np.mean(y)
            return np.full_like(y, np.clip(p, 0.1, 0.9))
        elif family == "poisson":
            return np.clip(y, 0.1, None)
        elif family == "gamma":
            return np.clip(y, 0.1, None)
        else:
            return y.copy()

    # ── Link function: mu -> eta ──

    @staticmethod
    def _link_fn(mu, link):
        if link == "identity":
            return mu.copy()
        elif link == "log":
            return np.log(np.clip(mu, 1e-15, None))
        elif link == "logit":
            return np.log(np.clip(mu, 1e-15, None) / np.clip(1 - mu, 1e-15, None))
        elif link == "probit":
            return sp_stats.norm.ppf(np.clip(mu, 1e-15, 1 - 1e-15))
        elif link == "inverse":
            return 1.0 / np.clip(mu, 1e-15, None)
        else:
            raise ValueError(f"Unknown link: {link}")

    # ── Inverse link: eta -> mu ──

    @staticmethod
    def _inv_link_fn(eta, link):
        if link == "identity":
            return eta.copy()
        elif link == "log":
            return np.exp(np.clip(eta, -500, 500))
        elif link == "logit":
            eta_c = np.clip(eta, -500, 500)
            return 1.0 / (1.0 + np.exp(-eta_c))
        elif link == "probit":
            return sp_stats.norm.cdf(np.clip(eta, -20, 20))
        elif link == "inverse":
            return 1.0 / np.clip(eta, 1e-15, None)
        else:
            raise ValueError(f"Unknown link: {link}")

    # ── d(eta)/d(mu) ──

    @staticmethod
    def _d_eta_d_mu(mu, link):
        if link == "identity":
            return np.ones_like(mu)
        elif link == "log":
            return 1.0 / np.clip(mu, 1e-15, None)
        elif link == "logit":
            return 1.0 / np.clip(mu * (1 - mu), 1e-15, None)
        elif link == "probit":
            # d(probit^{-1}(mu))/d(mu) via chain rule: d_eta/d_mu = 1/phi(eta)
            # but for IRLS we need d_eta/d_mu = 1/phi(Phi^{-1}(mu))
            eta = sp_stats.norm.ppf(np.clip(mu, 1e-15, 1 - 1e-15))
            phi = sp_stats.norm.pdf(eta)
            return 1.0 / np.clip(phi, 1e-15, None)
        elif link == "inverse":
            return -1.0 / np.clip(mu ** 2, 1e-15, None)
        else:
            raise ValueError(f"Unknown link: {link}")

    # ── Variance function V(mu) ──

    @staticmethod
    def _variance_fn(mu, family):
        if family == "gaussian":
            return np.ones_like(mu)
        elif family == "binomial":
            return np.clip(mu * (1 - mu), 1e-15, None)
        elif family == "poisson":
            return np.clip(mu, 1e-15, None)
        elif family == "gamma":
            return np.clip(mu ** 2, 1e-15, None)
        else:
            raise ValueError(f"Unknown family: {family}")

    # ── Clamp mu ──

    @staticmethod
    def _clamp_mu(mu, family):
        if family == "binomial":
            return np.clip(mu, 1e-10, 1 - 1e-10)
        elif family == "poisson":
            return np.clip(mu, 1e-10, None)
        elif family == "gamma":
            return np.clip(mu, 1e-10, None)
        return mu

    # ── Log-likelihood ──

    @staticmethod
    def _log_likelihood(y, mu, family):
        mu = np.clip(mu, 1e-15, None)
        if family == "gaussian":
            # ll = -n/2 * log(2*pi*sigma2) - 1/(2*sigma2) * sum((y-mu)^2)
            # Use sigma2 = 1 (canonical) for quasi-ll comparison
            # Actually for deviance-based ll: use normal ll with MLE sigma2
            ss = np.sum((y - mu) ** 2)
            n = len(y)
            sigma2 = ss / n
            if sigma2 < 1e-15:
                sigma2 = 1e-15
            ll = -n / 2 * np.log(2 * np.pi * sigma2) - ss / (2 * sigma2)
            return float(ll)
        elif family == "binomial":
            return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu + 1e-15)))
        elif family == "poisson":
            # ll = sum(y*log(mu) - mu - log(y!))
            # For pseudo-r2 comparison the -log(y!) cancels, but we keep it
            from scipy.special import gammaln
            return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
        elif family == "gamma":
            # ll = sum((a-1)*log(y) - y/(mu/a) - a*log(a*mu) - log(Gamma(a)))
            # Use shape a estimated from data
            from scipy.special import gammaln
            a = np.mean(mu) ** 2 / (np.var(y - mu) + 1e-15)
            a = max(a, 0.1)
            ll = np.sum((a - 1) * np.log(np.clip(y, 1e-15, None))
                        - y * a / np.clip(mu, 1e-15, None)
                        - a * np.log(np.clip(mu, 1e-15, None))
                        - gammaln(a))
            return float(ll)
        else:
            raise ValueError(f"Unknown family: {family}")

    # ── Null model log-likelihood ──

    @staticmethod
    def _fit_null_model(y, family, link, n):
        """Fit intercept-only model for ll_0."""
        if family == "gaussian":
            mu0 = np.mean(y)
            ss = np.sum((y - mu0) ** 2)
            sigma2 = ss / n
            if sigma2 < 1e-15:
                sigma2 = 1e-15
            ll_0 = -n / 2 * np.log(2 * np.pi * sigma2) - ss / (2 * sigma2)
            return float(ll_0)
        elif family == "binomial":
            p_bar = np.mean(y)
            p_bar = np.clip(p_bar, 1e-15, 1 - 1e-15)
            return float(np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar)))
        elif family == "poisson":
            from scipy.special import gammaln
            mu0 = np.mean(y)
            mu0 = max(mu0, 1e-15)
            return float(np.sum(y * np.log(mu0) - mu0 - gammaln(y + 1)))
        elif family == "gamma":
            from scipy.special import gammaln
            mu0 = np.mean(y)
            mu0 = max(mu0, 1e-15)
            a = mu0 ** 2 / (np.var(y) + 1e-15)
            a = max(a, 0.1)
            ll_0 = np.sum((a - 1) * np.log(np.clip(y, 1e-15, None))
                          - y * a / mu0
                          - a * np.log(mu0)
                          - gammaln(a))
            return float(ll_0)
        else:
            raise ValueError(f"Unknown family: {family}")

    # ── Deviance ──

    @staticmethod
    def _deviance(y, mu, family):
        mu = np.clip(mu, 1e-15, None)
        y_safe = np.clip(y, 1e-15, None)
        if family == "gaussian":
            return float(np.sum((y - mu) ** 2))
        elif family == "binomial":
            return float(2 * np.sum(
                y * np.log(y_safe / mu) + (1 - y) * np.log((1 - y + 1e-15) / (1 - mu + 1e-15))
            ))
        elif family == "poisson":
            from scipy.special import gammaln
            return float(2 * np.sum(y * np.log(y_safe / mu) - (y - mu)))
        elif family == "gamma":
            return float(2 * np.sum(-np.log(y_safe / mu) + (y - mu) / mu))
        else:
            raise ValueError(f"Unknown family: {family}")
