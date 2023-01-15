from typing import Optional, Union, List
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import numpy as np


class Brownian_Motion:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)

    def get_dW(self, intervals: int) -> np.ndarray:

        return self.generator.normal(loc=0.0, scale=1.0, size=intervals)

    def get_W(self, intervals: int) -> np.ndarray:

        dW = self.get_dW(intervals)
        dW_cs = dW.cumsum()
        return dW_cs

    def _get_correlated_dW(self, dW: np.ndarray, rho: float) -> np.ndarray:

        dW2 = self.get_dW(len(dW))

        if np.array_equal(dW2, dW):
            raise ValueError(
                "Brownian Increment error, try choosing different random state."
            )

        return rho * dW + np.sqrt(1 - rho**2) * dW2

    def get_corr_dW_matrix(
        self,
        intervals: int,
        n_procs: int,
        rho: Optional[float] = None,
    ) -> np.ndarray:

        dWs: list[np.ndarray] = []
        for i in range(n_procs):

            if i == 0 or rho is None:
                dW_i = self.get_dW(intervals)
            else:
                dW_corr_ref = self._get_corr_ref_dW(dWs, i)
                dW_i = self._get_correlated_dW(dW_corr_ref, rho)

            dWs.append(dW_i)

        return np.asarray(dWs).T

    def _get_corr_ref_dW(self, dWs: list[np.ndarray], i: int) -> np.ndarray:

        random_proc_idx = self.generator.choice(i)
        return dWs[random_proc_idx]


@dataclass
class OU_Params:

    mean_reversion: float
    asymptotic_mean: float
    std_dev: float


class OU_Process:
    def get_OU_process(
        self,
        intervals: int,
        OU_params: OU_Params,
        dW: np.array,
        X_0: Optional[float] = None,
    ) -> np.array:

        interval_arr = np.arange(intervals, dtype=np.longdouble)
        exp_alpha_t = np.exp(-OU_params.mean_reversion * interval_arr)

        integral_W = OU_Process._get_integal_W(interval_arr, dW, OU_params)
        _X_0 = OU_Process._select_X_0(X_0, OU_params)

        return (
            _X_0 * exp_alpha_t
            + OU_params.asymptotic_mean * (1 - exp_alpha_t)
            + OU_params.std_dev * exp_alpha_t * integral_W
        )

    def estimate_OU_params(self, X_t: np.ndarray) -> OU_Params:

        y = np.diff(X_t)
        X = X_t[:-1].reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)

        alpha = -reg.coef_[0]
        gamma = reg.intercept_ / alpha

        y_hat = reg.predict(X)
        beta = np.std(y - y_hat)

        return OU_Params(mean_reversion=alpha, asymptotic_mean=gamma, std_dev=beta)

    def get_corr_OU_procs(
        self,
        intervals: int,
        OU_params: Union[OU_Params, tuple[OU_Params, ...]],
        brownian_motion_instance: Brownian_Motion,
        n_procs: Optional[int] = None,
        proc_correlation: Optional[float] = None,
    ) -> np.ndarray:

        _n_procs = self._get_n_procs(OU_params, n_procs)
        corr_dWs = brownian_motion_instance.get_corr_dW_matrix(
            intervals, _n_procs, proc_correlation
        )

        OU_procs = []
        for i in range(_n_procs):

            if isinstance(OU_params, list):
                OU_params_i = OU_params[i]
            else:
                OU_params_i = OU_params

            dW_i = corr_dWs[:, i]

            ou_sim = self.get_OU_process(intervals, OU_params_i, dW_i)
            if any(np.isnan(ou_sim)):
                raise ValueError(f"{OU_params_i}, {i}/{_n_procs} had NAs. Failing")

            OU_procs.append(ou_sim)

        return np.asarray(OU_procs).T

    def _get_n_procs(
        self, OU_params: Union[OU_Params, List[OU_Params]], n_procs: Optional[int]
    ) -> int:

        if isinstance(OU_params, list):
            return len(OU_params)
        elif n_procs is None:
            raise ValueError("If OU_params is not tuple, n_procs must be specified")
        return n_procs

    def _select_X_0(X_0_in: Optional[float], OU_params: OU_Params) -> float:
        if X_0_in is not None:
            return X_0_in
        return OU_params.asymptotic_mean

    def _get_integal_W(
        intervals: np.ndarray, dW: np.ndarray, OU_params: OU_Params
    ) -> np.ndarray:

        exp_alpha_s = np.exp(OU_params.mean_reversion * intervals)
        integral_W = np.cumsum(exp_alpha_s * dW)
        return integral_W
