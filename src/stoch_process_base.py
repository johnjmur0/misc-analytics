from typing import Optional

import numpy as np


class Stochastic_Process:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)

    def get_dW(self, intervals: int) -> np.ndarray:

        return self.generator.normal(loc=0.0, scale=1.0, size=intervals)

    def get_W(self, intervals: int) -> np.ndarray:

        dW = Stochastic_Process.get_dW(self, intervals)
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
