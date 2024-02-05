# Lets test what the constant_idx method can resolve

import numpy as np
from functions.core_functions import mean_autocorrelation, pval_mac
from functions.general_functions import simulate_rectangular
import itertools as it
import time as time_module
import os

# ---------------------------------------------- #
# running index for parallelization
# ---------------------------------------------- #
cl_idx = 0  # int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", cl_idx, "type", type(cl_idx))
t = time_module.time()

# ---------------------------------------------- #
# fixed parameters
# ---------------------------------------------- #

n = 2  # number of times the magnitudesa are simulated to get the statistics

mc = 0
delta_m = 0.1

b = 1
n_bs = 200

b_parameter = "b_value"

cutting = "constant_idx"
transform = False

n_total = 10000
anomaly_func = "step"  # "sinus", "gaussian"

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #

delta_bs = np.arange(0.3, 1.1, 0.1)
n_series = np.arange(50, 1000, 50)
n_anomals = np.arange(100, 500, 100)

all_permutations = [
    i
    for i in it.product(
        delta_bs,
        n_series,
        n_anomals,
    )
]
all_permutations = np.array(all_permutations)

# parameter vectors to run through with cl_idx
cl_delta_bs = all_permutations[:, 0]
cl_n_series = all_permutations[:, 1].astype(int)
cl_n_anomals = all_permutations[:, 2].astype(int)

# -----------------------------------------------#
# simulate magnitudes and calculate acf
# -----------------------------------------------#

acf_mean = []
n_used_mean = []

for ii in range(n):
    mags, _ = simulate_rectangular(
        n_total, cl_n_anomals[cl_idx], b, cl_delta_bs[cl_idx], mc, delta_m
    )

    times = np.random.rand(len(mags)) * 1000
    times = np.sort(times)

    acfs, std_acf, n_series_used = mean_autocorrelation(
        mags,
        times,
        cl_n_series[cl_idx],
        mc=mc,
        delta_m=delta_m,
        n=500,
        transform=transform,
        cutting=cutting,
        b_method="tinti",
    )
    acf_mean.append(np.mean(acfs))
    n_used_mean.append(np.mean(n_series_used))
acf_mean = np.array(acf_mean)
n_used_mean = np.array(n_used_mean)

# -----------------------------------------------#
# find the significance level for the acf
# -----------------------------------------------#

p_val = pval_mac(acf_mean, n_used_mean, cutting)
p_val = np.array(p_val)

# -----------------------------------------------#
# save acfs
# -----------------------------------------------#

save_str = (
    "results/resolution/" + str(cutting) + "/p_val" + str(cl_idx) + ".csv"
)

# np.savetxt(save_str, p_val, delimiter=",")

print("time = ", time_module.time() - t)
