import comp_time_in_some_method as ct
import functionfile as ff
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')


n = 100
m = 10000
r = 50
approximate_size = 1000
program_num = "test"
# program_num = "1"
v_seed = 0   # base is 0
wh_seed = 1  # base is 1
c_mode = 0  # 0:MU NMF  1:HALS NMF
mode = 0  # 0:evaluate by Vo-V Frobenius norm
CDC = 5000  # base is 5000
time_ave_size = 10
convergence_D_mode = 1  # 0:non-seting   1:absolute convergence
NMFQP = False  # witch is NMF matrix H is calculated by QP or not.


# preparation ----------------------------------------------------------------------------------------------------------
print("-----------------   START   ------------------")

program_code = ff.program_name(mode, c_mode, n, m, r, approximate_size, "_unknown", v_seed, wh_seed, program_num,
                               convergence_D_mode=convergence_D_mode, CDC=CDC)
seeds = ff.two_seeds(wh_seed)

print("mode = " + str(mode) + "\nstart  " + program_code + "  >>>>>\n")
print("seed of V=" + str(v_seed) + "  seed of WH=" + str(wh_seed))

# generate V------------------------------------------------------------------------------------------------------------
V, original_W, original_H = ff.generate_wh(n, m, r, v_seed)

# get iteration---------------------------------------------------------------------------------------------------------
print("\n\n\n--------------  get iteration  --------------")

if mode == 0:
    iteration_list = ct.get_v_ite(V, n, m, r, approximate_size, seeds, c_mode, CDC, NMFQP=NMFQP)
else:
    print("ModeError : mode " + str(mode) + " is not define", file=sys.stderr)
    sys.exit(1)

print("\n\nfinish  |  NMF iteration : " + str(iteration_list[0]) + "   SNMF iteration : " + str(iteration_list[1]))


# time measurement -----------------------------------------------------------------------------------------------------
print("\n\n\n--------------  time measurement  --------------")
ite = "_NMF:" + str(iteration_list[0]) + " SNMF:" + str(iteration_list[1])
program_code = ff.program_name(mode, c_mode, n, m, r, approximate_size, ite, v_seed, wh_seed, program_num,
                               convergence_D_mode=convergence_D_mode, CDC=CDC)
print("\nstart  " + program_code + "  >>>>>\n")

t_result = np.zeros([2, time_ave_size])
for j in range(time_ave_size):
    for i in range(2):
        t_result[i, j] = ct.time_measurement(n, m, r, approximate_size, V, iteration_list[i], seeds, c_mode, i)

# save result ----------------------------------------------------------------------------------------------------------
s_program_code = ff.ct_save_program_name(mode, c_mode, ite, program_num, convergence_D_mode, CDC)

path = "../mnt/workspace/sNMF/randomM/comp_meth/"
os.makedirs(path + "/" + program_num, exist_ok=True)

d_t_result = pd.DataFrame({"NMF time": np.mean(t_result[0, :]), "SNMF time": np.mean(t_result[1, :]),
                           "NMF iteration": iteration_list[0], "SNMF iteration": iteration_list[1]}, index=[0])
d_t_result.to_csv(path + "/" + program_num + "/" + s_program_code + ".csv")
