import copy
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from src.stoch_process_base import (
    Brownian_Motion,
    OU_Process,
    OU_Params,
    CIR_Process,
    CIR_Params,
)


class Test_Brownian_Motion:
    def test_brownian_motion_change(self):

        stoch_instance = Brownian_Motion(seed=None)

        process = stoch_instance.get_dW(intervals=10000)

        assert len(process) == 10000
        # Abs b/c mean is 0
        assert process.mean() == pytest.approx(0, abs=0.01)
        assert process.std() == pytest.approx(1, rel=0.1)

    def test_brownian_motion_basic(self):

        stoch_instance = Brownian_Motion(seed=None)

        process = stoch_instance.get_W(intervals=100)

        assert len(process) == 100
        assert len(process[np.isnan(process)]) == 0

    def test_brownian_motion_seed(self):

        seed_same = 10
        stoch_instance = Brownian_Motion(seed=seed_same)
        process_1 = stoch_instance.get_W(intervals=100)

        stoch_instance_2 = Brownian_Motion(seed=seed_same)
        process_2 = stoch_instance_2.get_W(intervals=100)

        assert all(process_1 == process_2)

        stoch_instance_3 = Brownian_Motion(seed=seed_same * 2)
        process_3 = stoch_instance_3.get_W(intervals=100)

        assert process_3.sum() != process_1.sum()

    def test_brownian_motion_corr(self):

        seed_same = 10
        stoch_instance = Brownian_Motion(seed=seed_same)
        org_process = stoch_instance.get_dW(intervals=100)

        corr_val = 0.5
        corr_process_1 = stoch_instance._get_correlated_dW(org_process, corr_val)

        assert org_process.mean() != corr_process_1.mean()

        corr_val = 1
        corr_process_2 = stoch_instance._get_correlated_dW(org_process, corr_val)

        assert all(org_process == corr_process_2)

        corr_val = -1
        corr_process_3 = stoch_instance._get_correlated_dW(org_process, corr_val)
        all(org_process * -1 == corr_process_3)

    def test_brownian_multiple_corr(self):

        seed = 123
        stoch_instance = Brownian_Motion(seed=seed)

        corr_matrix_closer = stoch_instance.get_corr_dW_matrix(
            intervals=100, n_procs=50, rho=0.9
        )

        avg_variance_closer = (
            pd.DataFrame(np.cumsum(corr_matrix_closer, axis=0))
            .reset_index(drop=False)
            .melt(id_vars=["index"])
            .groupby(["index"])
            .var()
            .mean()[0]
        )

        corr_matrix_farther = stoch_instance.get_corr_dW_matrix(
            intervals=100, n_procs=50, rho=0.1
        )

        avg_variance_farther = (
            pd.DataFrame(np.cumsum(corr_matrix_farther, axis=0))
            .reset_index(drop=False)
            .melt(id_vars=["index"])
            .groupby(["index"])
            .var()
            .mean()[0]
        )

        assert avg_variance_closer < avg_variance_farther


class Test_OU_Process:
    def get_avg_corr(sim_arr):
        return (
            pd.DataFrame(np.corrcoef(np.diff(sim_arr, axis=0), rowvar=False))
            .reset_index(drop=False)
            .melt(id_vars=["index"])
            .groupby("variable")
            .agg({"value": "mean"})["value"]
            .mean()
        )

    def test_get_ou_process(self):

        intervals = 1000
        ou_proc = OU_Process()
        ou_params = OU_Params(mean_reversion=0.07, asymptotic_mean=0.01, std_dev=0.001)

        brw_motion = Brownian_Motion(seed=12345)
        dW = brw_motion.get_dW(intervals)

        ou_sim = ou_proc.get_OU_process(intervals, ou_params, dW)

        assert len(ou_sim) == intervals

        assert ou_sim[0] == pytest.approx(ou_params.asymptotic_mean, rel=0.2)
        assert ou_sim.mean() == pytest.approx(ou_params.asymptotic_mean, rel=0.1)
        assert ou_sim.std() == pytest.approx(ou_params.std_dev, rel=2)

    def test_get_ou_estimation(self):

        intervals = 1000
        ou_proc = OU_Process()
        ou_params = OU_Params(mean_reversion=0.1, asymptotic_mean=0.2, std_dev=0.05)

        brw_motion = Brownian_Motion(seed=6789)
        dW = brw_motion.get_dW(intervals)

        ou_sim = ou_proc.get_OU_process(intervals, ou_params, dW)

        ou_params_est = ou_proc.estimate_OU_params(ou_sim)

        assert ou_params_est.mean_reversion == pytest.approx(
            ou_params.mean_reversion, rel=0.25
        )
        assert ou_params_est.asymptotic_mean == pytest.approx(
            ou_params.asymptotic_mean, rel=0.05
        )
        assert ou_params_est.std_dev == pytest.approx(ou_params.std_dev, rel=0.05)

    def test_ou_corr_single(self):

        intervals = 1000
        brw_motion = Brownian_Motion(seed=91234)
        ou_proc = OU_Process()
        ou_params = OU_Params(mean_reversion=0.4, asymptotic_mean=4, std_dev=3)

        corr = 0.9
        n_proc = 10

        ou_sims = ou_proc.get_corr_OU_procs(
            intervals, ou_params, brw_motion, n_proc, corr
        )

        assert (intervals, n_proc) == ou_sims.shape

        larger_corr = Test_OU_Process.get_avg_corr(ou_sims)

        ou_sims_less_corr = ou_proc.get_corr_OU_procs(
            intervals, ou_params, brw_motion, n_proc, corr / 2
        )

        smaller_corr = Test_OU_Process.get_avg_corr(ou_sims_less_corr)

        assert larger_corr > smaller_corr

    def test_ou_corr_multiple(self):

        intervals = 1000
        brw_motion = Brownian_Motion(seed=91234)
        ou_proc = OU_Process()

        ou_param_list = [OU_Params(mean_reversion=0.1, asymptotic_mean=4, std_dev=3)]
        for i in np.arange(0.05, 0.25, 0.05):

            new_param = copy.deepcopy(ou_param_list[0])

            new_param.mean_reversion += i
            new_param.asymptotic_mean += i
            new_param.std_dev += i

            ou_param_list.append(new_param)

        corr = 0.9

        ou_sims = ou_proc.get_corr_OU_procs(intervals, ou_param_list, brw_motion, corr)

        assert (intervals, len(ou_param_list)) == ou_sims.shape

        # TODO don't remember where I was going with this test?
        multi_param_corr = Test_OU_Process.get_avg_corr(ou_sims)


class Test_CIR_Process:
    def test_single_sim(self):

        intervals = 1000
        CIR_proc = CIR_Process()
        CIR_params = CIR_Params(
            mean_reversion=0.06, asymptotic_mean=0.01, std_dev=0.009
        )
        brwn_inst = Brownian_Motion(12345)

        cir_sims = CIR_proc.get_CIR_process(intervals, CIR_params, brwn_inst)

        assert len(cir_sims) == intervals

        assert cir_sims[0] == pytest.approx(CIR_params.asymptotic_mean, rel=0.01)
        assert cir_sims.mean() == pytest.approx(CIR_params.asymptotic_mean, rel=0.01)
        assert cir_sims.std() == pytest.approx(CIR_params.std_dev, rel=.75)

    def test_estimate_cir_params(self):

        intervals = 1000
        CIR_proc = CIR_Process()
        CIR_params = CIR_Params(mean_reversion=0.05, asymptotic_mean=0.5, std_dev=0.02)
        brwn_inst = Brownian_Motion(12345)

        cir_sim = CIR_proc.get_CIR_process(intervals, CIR_params, brwn_inst)

        cir_param_est = CIR_proc.estimate_CIR_params(cir_sim)

        assert cir_param_est.mean_reversion == pytest.approx(
            CIR_params.mean_reversion, rel=0.35
        )
        assert cir_param_est.asymptotic_mean == pytest.approx(
            CIR_params.asymptotic_mean, rel=0.01
        )
        assert cir_param_est.std_dev == pytest.approx(CIR_params.std_dev, rel=0.05)
