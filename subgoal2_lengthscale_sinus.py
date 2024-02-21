# Lets test what the constant_idx method can resolve

import numpy as np
import pandas as pd
from functions.general_functions import simulate_randomfield, b_any_series
from functions.eval_functions import mac_different_n
from seismostats.analysis.estimate_beta import estimate_b_tinti
import itertools as it
import time as time_module
import os

# ---------------------------------------------- #
# running index for parallelization
# ---------------------------------------------- #
cl_idx = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", cl_idx, "type", type(cl_idx))
t = time_module.time()

# ---------------------------------------------- #
# fixed parameters
# ---------------------------------------------- #
n = 50  # number of times the magnitudesa are simulated to get the statistics

mc = 0
delta_m = 0.1

b = 1

b_parameter = "b_value"

cutting = "constant_idx"
transform = False

n_total = 80000

n_bs = np.arange(20, 4000, 20)

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #

delta_bs = np.arange(0.02, 0.22, 0.02)
length_scales = np.arange(100, 2000, 150)

all_permutations = [
    i
    for i in it.product(
        delta_bs,
        length_scales,
    )
]
all_permutations = np.array(all_permutations)

# parameter vectors to run through with cl_idx
cl_delta_bs = all_permutations[:, 0]
cl_length_scales = all_permutations[:, 1].astype(int)

# -----------------------------------------------#
# simulate magnitudes and calculate acf
# -----------------------------------------------#

n_series_list = n_total / n_bs

acfs_all = []
sim_numbers_all = []
ig_next1 = []  # information gain
ig_next = []
ig_here = []
n_bs_all = []

for ii in range(n):
    mags, b_true = simulate_randomfield(
        n_total, cl_length_scales[cl_idx], b, cl_delta_bs[cl_idx], mc, delta_m
    )

    times = np.arange(len(mags))

    # acf
    acfs, acf_std, n_bs, n_series_used = mac_different_n(
        mags,
        np.arange(len(mags)),
        mc,
        delta_m,
        n_bs=n_bs,
        cutting="constant_idx",
        transform=False,
        b_method="tinti",
        plotting=False,
    )
    n_bs = n_bs.astype(int)

    acfs_all.extend(acfs)
    sim_numbers_all = sim_numbers_all + [ii] * len(acfs)
    n_bs_all.extend(n_bs)

    for n_b in n_bs:
        # measure how well the signal is reconstructed with the different n_bs
        b_time, idx_max, b_std = b_any_series(
            mags,
            times,
            n_b=n_b,
            mc=mc,
            delta_m=delta_m,
            return_std=True,
            overlap=0.95,
            method="tinti",
        )
        b_all = estimate_b_tinti(mags, mc, delta_m)

        # 1. information gain at the next time step
        idx_del = idx_max < len(b_true) - 1
        idx_max1 = idx_max[idx_del]
        inf_gain_next1 = (
            np.log(b_time[idx_del] / b_all)
            - (b_time[idx_del] - b_all) / b_true[idx_max1 + 1]
        )
        ig_next1.append(np.mean(inf_gain_next1))

        # 2. information gain at the next n_b time steps
        idx_del = idx_max < len(b_true) - n_b
        idx_max2 = idx_max[idx_del]
        inf_gain_next = np.zeros(len(b_true[idx_max2 + 1]))
        for ii in range(1, n_b):
            inf_gain_next += (
                np.log(b_time[idx_del] / b_all)
                - (b_time[idx_del] - b_all) / b_true[idx_max2 + ii]
            )
        ig_next.append(np.mean(inf_gain_next) / n_b)

        # 3. information gain at the times from which the signal is
        # reconstructed
        half = int(n_b / 2)
        inf_gain_here = (
            np.log(b_time / b_all) - (b_time - b_all) / b_true[idx_max - half]
        )
        ig_here.append(np.mean(inf_gain_here))

acfs_all = np.array(acfs_all)
sim_numbers_all = np.array(sim_numbers_all)
n_bs = np.array(n_bs_all)
ig_here = np.array(ig_here)
ig_next = np.array(ig_next)

# make dataframe
df = pd.DataFrame(
    {
        "acf": acfs_all,
        "sim_number": sim_numbers_all,
        "n_b": n_bs_all,
        "ig_here": ig_here,
        "ig_next": ig_next,
        "ig_next1": ig_next1,
    }
)

# -----------------------------------------------#
# save acfs
# -----------------------------------------------#

save_str = (
    "results/length_scale/" + str(cutting) + "/df" + str(cl_idx) + ".csv"
)
df.to_csv(save_str, sep=",", index=False)

print("time = ", time_module.time() - t)
