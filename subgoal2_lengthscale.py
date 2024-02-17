# Lets test what the constant_idx method can resolve

import numpy as np
from functions.core_functions import mean_autocorrelation
from functions.general_functions import simulate_randomfield, b_any_series
from functions.eval_functions import mac_different_n
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
mc = 0
delta_m = 0.1

b = 1

b_parameter = "b_value"

cutting = "constant_idx"
transform = False

n_total = 40000
anomaly_func = "gaussian"

n_bs = np.arange(20, 1500, 10)

# ---------------------------------------------- #
# varying parameters
# ---------------------------------------------- #

delta_bs = np.arange(0.05, 0.3, 0.1)
length_scales = np.arange(100, 2000, 150)
sim_numbers = np.arange(0, 50, 1)

all_permutations = [
    i
    for i in it.product(
        delta_bs,
        sim_numbers,
        length_scales,
    )
]
all_permutations = np.array(all_permutations)

# parameter vectors to run through with cl_idx
cl_delta_bs = all_permutations[:, 0]
cl_sim_numbers = all_permutations[:, 1].astype(int)
cl_length_scales = all_permutations[:, 2].astype(int)

# -----------------------------------------------#
# simulate magnitudes and calculate acf
# -----------------------------------------------#

n_series_list = n_total / n_bs

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

diff_one = []
diff_nb = []

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

    # 1. difference of the b-value estimate to the true b-value at the next time step
    idx_del = idx_max < len(b_true) - 1
    idx_max_1 = idx_max[idx_del]
    diff = (
        (b_time[idx_del] - b_true[idx_max_1 + 1]) / b_true[idx_max_1 + 1]
    ) ** 2
    diff_one.append(sum(diff) / len(diff))

    # 2. difference of the b-value estimate to the mean true b-value of the next n_b time steps
    b_expected = np.convolve(b_true, np.ones(n_b) / n_b, mode="valid")
    idx_del = idx_max < (len(b_true) - n_b)
    idx_max_nb = idx_max[idx_del]
    diff = (
        abs(
            (b_time[idx_del] - b_expected[idx_max_nb]) / b_expected[idx_max_nb]
        )
        ** 2
    )
    diff_nb.append(sum(diff) / len(diff))

diff_one = np.array(diff_one)
diff_nb = np.array(diff_nb)

# -----------------------------------------------#
# save acfs
# -----------------------------------------------#

save_str = (
    "results/length_scale/" + str(cutting) + "/acfs" + str(cl_idx) + ".csv"
)
np.savetxt(save_str, acfs, delimiter=",")
np.savetxt(
    save_str.replace("acfs", "n_series_used"), n_series_used, delimiter=","
)
np.savetxt(save_str.replace("acfs", "diff_one"), diff_one, delimiter=",")
np.savetxt(save_str.replace("acfs", "diff_nb"), diff_nb, delimiter=",")

print("time = ", time_module.time() - t)
