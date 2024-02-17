# Lets test what the constant_idx method can resolve

import numpy as np
from functions.core_functions import mean_autocorrelation
from functions.general_functions import simulate_randomfield, b_any_series
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

n_total = 40000
anomaly_func = "gaussian"

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #

delta_bs = np.arange(0.05, 0.3, 0.1)
n_bs = np.arange(20, 1500, 10)
length_scales = np.arange(100, 2000, 150)


all_permutations = [
    i
    for i in it.product(
        delta_bs,
        n_bs,
        length_scales,
    )
]
all_permutations = np.array(all_permutations)

# parameter vectors to run through with cl_idx
cl_delta_bs = all_permutations[:, 0]
cl_n_bs = all_permutations[:, 1].astype(int)
cl_length_scales = all_permutations[:, 2].astype(int)

# -----------------------------------------------#
# simulate magnitudes and calculate acf
# -----------------------------------------------#

n_series = n_total / cl_n_bs[cl_idx]

acf_mean = []
n_used_mean = []
diff_one = []
diff_nb = []

for ii in range(n):
    mags, b_true = simulate_randomfield(
        n_total, cl_length_scales[cl_idx], b, cl_delta_bs[cl_idx], mc, delta_m
    )

    times = np.random.rand(len(mags)) * 1000
    times = np.sort(times)

    # acf
    acfs, std_acf, n_series_used = mean_autocorrelation(
        mags,
        times,
        n_series,
        mc=mc,
        delta_m=delta_m,
        n=500,
        transform=transform,
        cutting=cutting,
        b_method="tinti",
    )
    acf_mean.append(np.mean(acfs))
    n_used_mean.append(np.mean(n_series_used))

    # measure how well the signal is reconstructed with the different n_bs
    b_time, idx_max, b_std = b_any_series(
        mags,
        times,
        n_b=cl_n_bs[cl_idx],
        mc=mc,
        delta_m=delta_m,
        return_std=True,
        overlap=0.95,
        method="tinti",
    )

    # 1. difference of the b-value estimate to the true b-value at the next time step
    idx_del = idx_max < len(b_true) - 1
    idx_max_1 = idx_max[idx_del]
    diff = (
        (b_time[idx_del] - b_true[idx_max_1 + 1]) / b_true[idx_max_1 + 1]
    ) ** 2
    diff_one.append(sum(diff) / len(diff))

    # 2. difference of the b-value estimate to the mean true b-value of the next n_b time steps
    b_expected = np.convolve(
        b_true, np.ones(cl_n_bs[cl_idx]) / cl_n_bs[cl_idx], mode="valid"
    )
    idx_del = idx_max < (len(b_true) - cl_n_bs[cl_idx])
    idx_max_nb = idx_max[idx_del]
    diff = (
        abs(
            (b_time[idx_del] - b_expected[idx_max_nb]) / b_expected[idx_max_nb]
        )
        ** 2
    )
    diff_nb.append(sum(diff) / len(diff))

acf_mean = np.array(acf_mean)
n_used_mean = np.array(n_used_mean)
diff_one = np.array(diff_one)
diff_nb = np.array(diff_nb)

# -----------------------------------------------#
# save acfs
# -----------------------------------------------#

save_str = (
    "results/length_scale/" + str(cutting) + "/acf_mean" + str(cl_idx) + ".csv"
)
np.savetxt(save_str, acf_mean, delimiter=",")
np.savetxt(
    save_str.replace("acf_mean", "n_used_mean"), n_used_mean, delimiter=","
)
np.savetxt(save_str.replace("acf_mean", "diff_one"), diff_one, delimiter=",")
np.savetxt(save_str.replace("acf_mean", "diff_nb"), diff_nb, delimiter=",")

print("time = ", time_module.time() - t)
