from typing import Optional, Union, List, NoReturn, Any
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
        OU_params: Union[OU_Params, List[OU_Params]],
        brownian_motion_inst: Brownian_Motion,
        n_procs: Optional[int] = None,
        proc_correlation: Optional[float] = None,
    ) -> np.ndarray:

        _n_procs = self._get_n_procs(OU_params, n_procs)

        corr_dWs = brownian_motion_inst.get_corr_dW_matrix(
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


@dataclass
class CIR_Params(OU_Params):

    # mean_reversion: float
    # asymptotic_mean: float
    # std_dev: float

    # NOTE super fun, haven't seen post_init before!
    def __post_init__(self) -> Optional[NoReturn]:
        if 2 * self.mean_reversion * self.asymptotic_mean < self.std_dev**2:
            raise ValueError("2ab has to be less than or equal to c^2.")
        return None


class CIR_Process:
    def _validate_not_nan(self, dsigma_t: Any) -> Optional[NoReturn]:
        if np.isnan(dsigma_t):
            raise ValueError(
                "CIR process simulation crashed, check your CIR_params. "
                + "Maybe choose a smaller c value."
            )
        return None

    def get_CIR_process(
        self,
        intervals: int,
        CIR_params: CIR_Params,
        brownian_motion_inst: Brownian_Motion,
        sigma_0: Optional[float] = None,
    ) -> np.ndarray:

        dW = brownian_motion_inst.get_dW(intervals)
        return self._generate_CIR_process(dW, CIR_params, sigma_0)

    def _generate_CIR_process(
        self, dW: np.ndarray, CIR_params: CIR_Params, sigma_0: Optional[float] = None
    ) -> np.ndarray:

        if sigma_0 is None:
            sigma_0 = CIR_params.asymptotic_mean

        sigma_t = [sigma_0]
        for t in range(1, len(dW)):

            dsigma_t = (
                CIR_params.mean_reversion
                * (CIR_params.asymptotic_mean - sigma_t[t - 1])
                + CIR_params.std_dev * np.sqrt(sigma_t[t - 1]) * dW[t]
            )

            self._validate_not_nan(dsigma_t)
            sigma_t.append(sigma_t[t - 1] + dsigma_t)

        return np.asarray(sigma_t)

    def estimate_CIR_params(self, sigma_t: np.ndarray) -> CIR_Params:

        sigma_sqrt = np.sqrt(sigma_t[:-1])
        y = np.diff(sigma_t) / sigma_sqrt
        x1 = 1.0 / sigma_sqrt
        x2 = sigma_sqrt
        X = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)

        ab = reg.coef_[0]
        a = -reg.coef_[1]
        b = ab / a

        y_hat = reg.predict(X)
        c = np.std(y - y_hat)
        return CIR_Params(mean_reversion=a, asymptotic_mean=b, std_dev=c)

    # TODO this is so similar to OU process, abstract into base class once sure of differences
    def get_corr_CIR_procs(
        self,
        intervals: int,
        CIR_params: Union[CIR_Params, List[CIR_Params]],
        brownian_motion: Brownian_Motion,
        n_procs: Optional[int] = None,
        corr: Optional[float] = None,
    ) -> np.ndarray:

        _n_procs = self._get_n_procs(CIR_params, n_procs)

        corr_dWs = brownian_motion.get_corr_dW_matrix(intervals, _n_procs, corr)

        CIR_procs = []
        for i in range(_n_procs):

            if isinstance(CIR_params, list):
                CIR_params_i = CIR_params[i]
            else:
                CIR_params_i = CIR_params

            dW_i = corr_dWs[:, i]

            cir_sim = self._generate_CIR_process(dW_i, CIR_params_i)

            CIR_procs.append(cir_sim)

        return np.asarray(CIR_procs).T

    def _get_n_procs(
        self, CIR_params: Union[CIR_Params, List[CIR_Params]], n_procs: Optional[int]
    ) -> int:

        if isinstance(CIR_params, list):
            return len(CIR_params)
        elif n_procs is None:
            raise ValueError("If CIR_params is not tuple, n_procs cannot be None.")
        return n_procs