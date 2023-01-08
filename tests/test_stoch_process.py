import numpy as np
import pytest
from src.stoch_process_base import Stochastic_Process


class Test_Stochastic_Process:
    def test_brownian_motion_change(self):

        stoch_instance = Stochastic_Process(seed=None)

        process = stoch_instance.get_dW(intervals=10000)

        assert len(process) == 10000
        assert process.mean() == pytest.approx(0, abs=0.1)
        assert process.std() == pytest.approx(1, abs=0.1)

    def test_brownian_motion_basic(self):

        stoch_instance = Stochastic_Process(seed=None)

        process = stoch_instance.get_W(intervals=100)

        assert len(process) == 100
        assert len(process[np.isnan(process)]) == 0

    def test_brownian_motion_seed(self):

        seed_same = 10
        stoch_instance = Stochastic_Process(seed=seed_same)
        process_1 = stoch_instance.get_W(intervals=100)

        stoch_instance_2 = Stochastic_Process(seed=seed_same)
        process_2 = stoch_instance_2.get_W(intervals=100)

        assert all(process_1 == process_2)

        stoch_instance_3 = Stochastic_Process(seed=seed_same * 2)
        process_3 = stoch_instance_3.get_W(intervals=100)

        assert process_3.sum() != process_1.sum()

    def test_brownian_motion_corr(self):

        seed_same = 10
        stoch_instance = Stochastic_Process(seed=seed_same)
        org_process = stoch_instance.get_dW(intervals=100)

        corr_val = 0.5
        corr_process_1 = stoch_instance.get_correlated_dW(org_process, corr_val)

        assert org_process.mean() != corr_process_1.mean()

        corr_val = 1
        corr_process_2 = stoch_instance.get_correlated_dW(org_process, corr_val)

        assert all(org_process == corr_process_2)

        corr_val = -1
        corr_process_3 = stoch_instance.get_correlated_dW(org_process, corr_val)
        all(org_process * -1 == corr_process_3)

    def test_brownian_multiple_corr(self):

        seed = 123
        stoch_instance = Stochastic_Process(seed=seed)

        corr_matrix_closer = stoch_instance.get_corr_dW_matrix(
            intervals=100, n_procs=50, rho=0.75
        )

        closer_means = [x.mean() for x in corr_matrix_closer]

        corr_matrix_farther = stoch_instance.get_corr_dW_matrix(
            intervals=100, n_procs=50, rho=0.25
        )

        farther_means = [x.mean() for x in corr_matrix_farther]

        assert np.std(closer_means) < np.std(farther_means)
