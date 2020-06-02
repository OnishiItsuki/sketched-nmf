import functionfile as ff
import time_measurement as tm
import pivoting_qr_q_eval as qr_eval
import v_error_eval as v_eval
import least_square_w_eval as ls_eval
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


n = 100
m = 10000
r = 50
approximate_size = 1000
iteration = 5000
program_num = "14"
v_seed = 0   # base is 0
wh_seed = 1  # base is 1
c_mode = 0  # 0:MU NMF  1:HALS NMF
mode = 0    # 0:time measurement  1:evaluate by pivotQR Qo-Q Frobenius norm  2:evaluate by Vo-V Frobenius norm  3:lstsq

program_code = ff.program_name(mode, c_mode, n, m, r, approximate_size, iteration, v_seed, wh_seed, program_num)
seeds = ff.two_seeds(wh_seed)

# generate V------------------------
print("-----------------   START   ------------------")
print("mode = " + str(mode) + "\nstart  " + program_code + "  >>>>>\n")
print("seed of V=" + str(v_seed) + "  seed of WH=" + str(wh_seed))
V, original_W, original_H = ff.generate_wh(n, m, r, v_seed)

# calculate ----------------------------
if mode == 0:
    t_result, _, _, _, _ = tm.time_measurement(n, m, r, approximate_size, V, iteration, seeds, c_mode)
elif mode == 1:
    NMF_error, SNMF_error, _, _, _, _ = qr_eval.pivoting_qr_q_eval(n, m, r, approximate_size, V, iteration, seeds,
                                                                   c_mode, original_W)
elif mode == 2:
    NMF_error, SNMF_error, _, _, _, _ = v_eval.v_error_eval(n, m, r, approximate_size, V, iteration, seeds, c_mode)
elif mode == 3:
    NMF_error, SNMF_error, _, _, _, _ = ls_eval.least_square_w_eval(n, m, r, approximate_size, V, iteration, seeds,
                                                                    c_mode, original_W)

# plot result----------------------------
path = "../mnt/workspace/sketchingNMF/random matrix/"
os.makedirs(path + "graph/" + program_num, exist_ok=True)
os.makedirs(path + "error/" + program_num, exist_ok=True)
os.makedirs(path + "time/" + program_num, exist_ok=True)

if mode != 0:
    plt.plot(range(1, iteration + 1), NMF_error, label="NMF")
    plt.plot(range(1, iteration + 1), SNMF_error, label="Sketching")
    plt.xlabel("the number of iteration")
    plt.ylabel("Frobenius norm")
    plt.title("error plot")
    plt.legend()
    plt.savefig(path + "graph/" + program_num + "/" + program_code + ".png")
    plt.close()

    plt.plot(range(1, iteration + 1), SNMF_error - NMF_error)
    plt.xlabel("the number of iteration")
    plt.ylabel("error difference")
    plt.title("The difference between Sketching NMF error and NMF error")
    plt.savefig(path + "graph/" + program_num + "/Dif" + program_code + ".png")

if mode != 0:
    e_result = pd.DataFrame([NMF_error, SNMF_error], index=["NMF error", "SNMF error"])
    e_result.T.to_csv(path + "error/" + program_num + "/" + program_code + "_error.csv")

if mode == 0:
    d_t_result = pd.DataFrame({"NMF time": t_result[0], "SNMF time": t_result[1]}, index=[0])
    d_t_result.to_csv(path + "time/" + program_num + "/" + program_code + "_time.csv")
