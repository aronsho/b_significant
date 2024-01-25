# In this file, tests for the resolution of the methods are performed. First,
# we check, wht eactly the distribution of the autocorrelation function is.
# Then, we test what can be resolved with the methods.
# same as Resolution_2.py but without transformation

import numpy as np
from functions.core_functions import get_acf_random
from functions.general_functions import simulated_magnitudes_binned
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
n = 500  # number of times the magnitudesa are simulated to get the statistics

mc = 0
delta_m = 0.1
# delta_m = 0.2
# delta_m = 0.01

b_parameter = "b_value"

# cutting = "random_idx"
# cutting = "constant_idx"
cutting = "random"
# transform = True
transform = False

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #
n_series = [15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 250, 300]
n_totals = [5000, 10000, 15000, 20000, 25000, 30000, 35000]
bs = [0.7, 1.0, 1.3]

all_permutations = [
    i
    for i in it.product(
        n_series,
        n_totals,
        bs,
    )
]
all_permutations = np.array(all_permutations)

# parameter vectors to run through with cl_idx
cl_n_series = all_permutations[:, 0].astype(int)
cl_n_totals = all_permutations[:, 1].astype(int)
cl_bs = all_permutations[:, 2]

# -----------------------------------------------#
# simulate magnitudes and calculate acf
# -----------------------------------------------#

acf_mean = []

for ii in range(n):
    mags = simulated_magnitudes_binned(
        cl_n_totals[cl_idx],
        cl_bs[cl_idx],
        mc,
        delta_m,
    )

    times = np.random.rand(len(mags)) * 1000
    times = np.sort(times)

    acfs, n_series_used = get_acf_random(
        mags,
        times,
        n_sample=cl_n_series[cl_idx],
        mc=mc,
        delta_m=delta_m,
        transform=transform,
        cutting=cutting,
        b_method="tinti",
    )
    acf_mean.append(np.mean(acfs))

# -----------------------------------------------#
# save acfs
# -----------------------------------------------#

acf_mean = np.array(acf_mean)


if cutting == "constant_idx":
    save_str = (
        "results/distributions/cutting_" + str(cutting) + "/"
        "acf_mean_" + str(cl_idx) + ".csv"
    )
else:
    save_str = (
        "results/distributions/cutting_"
        + str(cutting)
        + "/transform_"
        + str(transform)
        + "/"
        "acf_mean_" + str(cl_idx) + ".csv"
    )


np.savetxt(save_str, acf_mean, delimiter=",")

print("time = ", time_module.time() - t)
