# Lets test what the constant_idx method can resolve

import numpy as np
from functions.core_functions import mean_autocorrelation
from functions.general_functions import simulate_step
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

n = 500  # number of times the magnitudesa are simulated to get the statistics

mc = 0
delta_m = 0.1

b = 1
n_bs = 200

b_parameter = "b_value"

cutting = "constant_idx"
transform = False

n_series = 100

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #

n_totals = [
    1000,
    2000,
    3000,
    4000,
    5000,
    7000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    50000,
]
delta_bs = np.arange(0, 1.025, 0.025)

all_permutations = [
    i
    for i in it.product(
        n_totals,
        delta_bs,
    )
]
all_permutations = np.array(all_permutations)

# parameter vectors to run through with cl_idx
cl_n_totals = all_permutations[:, 0].astype(int)
cl_delta_bs = all_permutations[:, 1]

# -----------------------------------------------#
# simulate magnitudes and calculate acf
# -----------------------------------------------#

acf_mean = []
n_used_mean = []

for ii in range(n):
    mags, _ = simulate_step(
        cl_n_totals[cl_idx], b, cl_delta_bs[cl_idx], mc, delta_m
    )

    times = np.random.rand(len(mags)) * 1000
    times = np.sort(times)

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
acf_mean = np.array(acf_mean)
n_used_mean = np.array(n_used_mean)


# -----------------------------------------------#
# save acfs
# -----------------------------------------------#

save_str = (
    "results/resolution/" + str(cutting) + "/acf_mean" + str(cl_idx) + ".csv"
)
np.savetxt(save_str, acf_mean, delimiter=",")
np.savetxt(
    save_str.replace("acf_mean", "n_used_mean"), n_used_mean, delimiter=","
)

print("time = ", time_module.time() - t)
