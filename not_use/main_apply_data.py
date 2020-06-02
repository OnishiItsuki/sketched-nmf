import apply_data as ad
import functionfile as ff
import image_processing_function as ipf
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')


# r = 50
# approximate_size = 1000
iteration = 5000
# program_num = "5(r=50,k=1000-2429)"
# program_num = "9(for_thesis)"
program_num = "tmp"
# program_num = "test"
v_seed = 0   # base is 0
wh_seed = 7  # base is 1
c_mode = 0  # 0:MU NMF  1:HALS NMF
mode = 2   # 0:time measurement  1:evaluate by pivotQR Qo-Q Frobenius norm  2:evaluate by Vo-V Frobenius norm  3:lstsq
# use_data = "CBCL/train"
use_data = "YaleFD/faces"
machine = 1  # 0:local   1:merkatz   2-4:konev02-04   5-:lutz
column_sketching = False
NMFQP = False

pre_r_list = [25]
# pre_ap_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2429]
pre_ap_list = [7500]

# preparation ----------------------------------------------------------------------------------------------------------
r_list = []
ap_list = []

r_list = pre_r_list
ap_list = pre_ap_list

# r_list.append(pre_r_list[0])
# ap_list.append(pre_ap_list[1])

r_data_size = len(r_list)
ap_data_size = len(ap_list)

if c_mode == 0:
    c_method = "MU"
elif c_mode == 1:
    c_method = "HALS"

if machine == 0:
    r_path = use_data
    w_path = c_method
    os.makedirs(w_path, exist_ok=True)
else:
    r_path = "/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data
    w_path = "/home/ionishi/mnt/workspace/sketchingNMF/" + use_data + "/" + program_num + "/" + c_method

v, im_var, im_hol = ipf.read_pgm(r_path)
v = v.T
if mode == 0:
    # time measurement--------------------------------------------------------------------------------------------------
    nmf_t_result = np.zeros([r_data_size, ap_data_size])
    snmf_t_result = np.zeros([r_data_size, ap_data_size])

    for r_i, r in enumerate(r_list):
        for ap_i, approximate_size in enumerate(ap_list):
            pw_path = w_path + "/r,k/r=" + str(r) + "/k=" + str(approximate_size)
            t_result = ad.apply_data(v, r, approximate_size, iteration, program_num, v_seed, wh_seed, c_mode, mode,
                                     pw_path, NMFQP, column_sketching=column_sketching)
            nmf_t_result[r_i, ap_i] = t_result[0]
            snmf_t_result[r_i, ap_i] = t_result[1]

    nmf_t_result = np.mean(nmf_t_result, axis=1)
    df = pd.DataFrame({"k=" + str(ap_list[0]): snmf_t_result[0, 0]}, index=["r=" + str(r_list[0])])
    for i in range(1, r_data_size):
        r = r_list[i]
        df.loc["r=" + str(r)] = snmf_t_result[i, 0]
    for j in range(1, ap_data_size):
        k = ap_list[j]
        df["k=" + str(k)] = snmf_t_result[:, j]
    df["NMF"] = nmf_t_result

    df.to_csv(w_path + "/time(column;r, row;appr_size).csv")
    df.to_latex(w_path + "/time(column;r, row;appr_size).tex")

else:
    # calculate error --------------------------------------------------------------------------------------------------

    for r_i, r in enumerate(r_list):
        for ap_i, approximate_size in enumerate(ap_list):
            pw_path = w_path + "/r,k/r=" + str(r) + "/k=" + str(approximate_size)
            NMF_error, SNMF_error = ad.apply_data(v, r, approximate_size, iteration, program_num, v_seed, wh_seed,
                                                  c_mode, mode, pw_path, NMFQP, column_sketching=column_sketching)
