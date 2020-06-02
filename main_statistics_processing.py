import image_processing_function as ipf
import functionfile as ff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# program_num = "7(change seed no QP)"
# program_num = "8(change seed NMF QP)"
# program_num = "9(for_thesis)"
# program_num = "10(for_MIRU)"
program_num = "11(for_MIRU_FGD_bag_fix)"
# program_num = "15(for_graduate_thesis)"
# program_num = "16(for_graduate_thesis)"

# use_data = "CBCL/train"
use_data = "YaleFD/faces"
# use_data = "random_matrix"

# c_method = "MU"
# c_method = "HALS"
c_method = "FGD"
# c_method = "GCD"


w_program_num = program_num
# w_program_num = "test"

space = False

# pd.options.display.float_format = '{:.4e}'.format

# =================check this when error printed ================
seed_list = range(1, 11)

if "CBCL" in use_data:
    r_list = [50]
    ap_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2429]
elif "YaleFD" in use_data:
    # r_list = [15, 50]
    # ap_list = [20, 50, 80, 100, 130, 165]
    r_list = [25]
    ap_list = [1000, 2500, 5000, 7500, 10000, 30000, 50000, 77760]
cal_mse = True

if "MU" in c_method:
    ite = 3000
    error_r_ite = 5000
elif "HALS" in c_method:
    ite = 400
    error_r_ite = 500
else:
    ite = 400
    error_r_ite = 400

w_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/".format(use_data, w_program_num, c_method)
if "random" in use_data:
    r_list = [50]
    ap_list = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    cal_mse = False
    if "MU" in c_method:
        ite = 5000
    elif "HALS" in c_method:
        ite = 500
    error_r_ite = 5000
    w_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}_iteration{}/".format(use_data, w_program_num, c_method, ite)

if "test" in w_program_num:
    seed_list = [1]


r_list_size = len(r_list)
ap_list_size = len(ap_list)
seed_list_size = len(seed_list)

os.makedirs(w_path, exist_ok=True)

print("start  {}  {}  {} >>>>>".format(w_program_num, use_data, c_method))
# time measurement--------------------------------------------------------------------------------------------------

nmf_t_result = np.zeros([r_list_size, seed_list_size])
snmf_t_result = np.zeros([r_list_size, ap_list_size, seed_list_size])

for r_i, r in enumerate(r_list):
    for ap_i, ap in enumerate(ap_list):
        for seed_i, seed in enumerate(seed_list):
            pr_path, pr_file_name = ff.path_and_file_name(r, ap, ite, seed, use_data, program_num, c_method, "r,k",
                                                          directory="time", space=space)
            t_result = pd.read_csv(pr_path + pr_file_name + ".csv")
            snmf_t_result[r_i, ap_i, seed_i] = t_result["SNMF time"]
            if ap == ap_list[-1]:
                nmf_t_result[r_i, seed_i] = t_result["NMF time"]

nmf_t_result = np.mean(nmf_t_result, axis=1)
snmf_t_result = np.mean(snmf_t_result, axis=2)

df_t_result = pd.DataFrame({"k=" + str(ap_list[0]): snmf_t_result[0, 0]}, index=["r=" + str(r_list[0])])
for r_i, r in enumerate(r_list[1:]):
    df_t_result.loc["r=" + str(r)] = snmf_t_result[r_i + 1, 0]

for ap_i, k in enumerate(ap_list[1:]):
    df_t_result["k=" + str(k)] = snmf_t_result[:, ap_i + 1]
df_t_result["NMF"] = nmf_t_result

df_t_result.to_csv(w_path + "/time(ite{}).csv".format(ite))
# df_t_result.to_latex(w_path + "/time(ite{}).tex".format(ite))

# last 1%  of iteration error average  ---------------------------------------------------------------------------------
# NMF_e_result = np.zeros([r_list_size, seed_list_size])
# SNMF_e_result = np.zeros([r_list_size, ap_list_size, seed_list_size])
#
# for r_i, r in enumerate(r_list):
#     for ap_i, ap in enumerate(ap_list):
#         for seed_i, seed in enumerate(seed_list):
#             pr_path, pr_file_name = ff.path_and_file_name(r, ap, error_r_ite, seed, use_data, program_num, c_method, "r,k",
#                                                           directory="error", space=space)
#             error_M = pd.read_csv(pr_path + pr_file_name + "_error.csv")
#             e_result = error_M.iloc[ite - int(0.01 * ite):ite].mean()
#             SNMF_e_result[r_i, ap_i, seed_i] = e_result["SNMF error"]
#             if ap == ap_list[-1]:
#                 NMF_e_result[r_i, seed_i] = e_result["NMF error"]
#
# NMF_e_result = np.mean(NMF_e_result, axis=1)
# SNMF_e_result = np.mean(SNMF_e_result, axis=2)
#
# df_e_result = pd.DataFrame({"k=" + str(ap_list[0]): SNMF_e_result[0, 0]}, index=["r=" + str(r_list[0])])
# for r_i, r in enumerate(r_list[1:]):
#     df_e_result.loc["r=" + str(r)] = SNMF_e_result[r_i + 1, 0]
# for ap_i, k in enumerate(ap_list[1:]):
#     df_e_result["k=" + str(k)] = SNMF_e_result[:, ap_i + 1]
# df_e_result["NMF"] = NMF_e_result
#
# df_e_result.to_csv(w_path + "error_ave(ave of -1%~).csv")
# df_e_result.to_latex(w_path + "error_ave(ave of -1%~).tex")
# # ipf.checkerboard_table(df_e_result, w_path, "result of convergence(GT_NMF_SNMF).png")

# MSE  -----------------------------------------------------------------------------------------------------------------
if cal_mse:
    mse_ave_result = np.zeros([ap_list_size, seed_list_size, 3])  # [0]:ori-nmf  [1]:ori-snmf  [2]:nmf-snmf
    # mse_max_result = np.zeros([ap_list_size, seed_list_size, 3])  # [0]:ori-nmf  [1]:ori-snmf  [2]:nmf-snmf

    v_original, im_var, im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data)
    v_list = np.zeros([v_original.shape[0], v_original.shape[1], 3])  # [0]:original  [1]:snmf  [2]:nmf
    v_list[:, :, 0] = v_original
    for r_i, r in enumerate(r_list):
        for ap_i, ap in enumerate(ap_list):
            for seed_i, seed in enumerate(seed_list):
                pr_path, pr_file_name = ff.path_and_file_name(r, ap, ite, seed, use_data, program_num, c_method, "r,k",
                                                              directory="matrix", space=space)
                v_list[:, :, 1] = ipf.calculate_v(pr_path, pr_file_name, snmf=True)

                pr_path, pr_file_name = ff.path_and_file_name(r, ap_list[-1], ite, seed, use_data, program_num, c_method,
                                                              "r,k", directory="matrix", space=space)
                v_list[:, :, 2] = ipf.calculate_v(pr_path, pr_file_name, snmf=False)

                for i in range(3):
                    mse_ave_result[ap_i, seed_i, (i + 1) % 3]= \
                        ipf.mse_calculate(v_list[:, :, i], v_list[:, :, (i + 1) % 3])

        df_ave_mse = ipf.make_mse_dataframe(mse_ave_result, ap_list)
        # df_max_mse = ipf.make_mse_dataframe(mse_max_result, ap_list)

        df_ave_mse.to_csv(w_path + "r={}_ave_MSE(GT_NMF_SNMF).csv".format(r))
        # df_ave_mse.to_latex(w_path + "r={}_ave_MSE(GT_NMF_SNMF).tex".format(r))
        # df_max_mse.to_csv(w_path + "r={}_max_MSE(GT_NMF_SNMF).csv".format(r))
        # df_max_mse.to_latex(w_path + "r={}_max_MSE(GT_NMF_SNMF).tex".format(r))


