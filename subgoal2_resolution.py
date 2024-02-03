# Lets test what the constant_idx method can resolve

import numpy as np
from functions.core_functions import mean_autocorrelation, pval_mac
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

n_sample = 100
p = 0.05  # significance level for the acf

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #

n_totals = [5000, 10000, 15000, 20000, 25000, 30000, 35000]
delta_bs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

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
        n_sample,
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

np.savetxt(save_str, acf_mean, delimiter=",")
