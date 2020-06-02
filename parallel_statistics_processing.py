import image_processing_function as ipf
import functionfile as ff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def cal_three_mse(r_list, ap_list, seed_list, iteration, use_data, program_num, c_method):
    result = np.zeros([len(ap_list), len(seed_list), 3])  # [0]:ori-nmf  [1]:ori-snmf  [2]:nmf-snmf

    v_original, im_var, im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data)
    v_list = np.zeros([v_original.shape[0], v_original.shape[1], 3])  # [0]:original  [1]:snmf  [2]:nmf
    v_list[:, :, 0] = v_original

    for ap_i, ap in enumerate(ap_list):
        for seed_i, seed in enumerate(seed_list):
            path, file_name = ff.path_and_file_name(r_list[0], ap, iteration, seed, use_data, program_num, c_method,
                                                    "r,k", directory="matrix")

            v_list[:, :, 1] = ipf.calculate_v(path, file_name, snmf=True)
            v_list[:, :, 2] = ipf.calculate_v(path, file_name, snmf=False)

            for i in range(3):
                result[ap_i, seed_i, (i + 1) % 3] = ipf.mse_calculate(v_list[:, :, i], v_list[:, :, (i + 1) % 3])
    return result


if __name__ == "__main__":
    # 関数化
    program_num = "10(for_MIRU)"
    use_data = "CBCL/train"
    c_method = "FGD"
    w_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/".format(use_data, "test", c_method)
    ite = 400
    significant_fig = 3

    tmp = sys.argv
    mode = tmp[1]  # 0:last 1%  of iteration error average  1:MSE
    r_list, r_list_size, ap_list, ap_list_size, seed_list, seed_list_size = ff.parallel_make_list(tmp[2:])

    # r_list_size = len(r_list)
    # ap_list_size = len(ap_list)
    # seed_list_size = len(seed_list)

    os.makedirs(w_path, exist_ok=True)

    # last 1%  of iteration error average  -----------------------------------------------------------------------------
    NMF_e_result = np.zeros([r_list_size, ap_list_size, seed_list_size])
    SNMF_e_result = np.zeros([r_list_size, ap_list_size, seed_list_size])

    for r_i, r in enumerate(r_list):
        for ap_i, ap in enumerate(ap_list):
            for seed_i, seed in enumerate(seed_list):
                pr_path, pr_file_name = ff.path_and_file_name(r, ap, ite, seed, use_data, program_num, c_method, "r,k", directory="error")
                error_M = pd.read_csv(pr_path + pr_file_name + "_error.csv")
                e_result = error_M.iloc[int(-0.01 * ite):].mean()
                NMF_e_result[r_i, ap_i, seed_i] = e_result["NMF error"]
                SNMF_e_result[r_i, ap_i, seed_i] = e_result["SNMF error"]

    NMF_e_result = np.mean(NMF_e_result, axis=2)
    SNMF_e_result = np.mean(SNMF_e_result, axis=2)

    NMF_e_result = np.mean(NMF_e_result, axis=1)
    df_e_result = pd.DataFrame({"k=" + str(ap_list[0]): SNMF_e_result[0, 0]}, index=["r=" + str(r_list[0])])
    for r_i, r in enumerate(r_list[1:]):
        df_e_result.loc["r=" + str(r)] = SNMF_e_result[r_i + 1, 0]
    for ap_i, k in enumerate(ap_list[1:]):
        df_e_result["k=" + str(k)] = SNMF_e_result[:, ap_i + 1]
    df_e_result["NMF"] = NMF_e_result

    df_e_result.to_csv(w_path + "error_ave(ave of -1%~).csv")

    # MSE  -------------------------------------------------------------------------------------------------------------
    mse_result = cal_three_mse(r_list, ap_list, seed_list, ite, use_data, program_num, c_method)
    mse_result = np.mean(mse_result, axis=1)

    df_mse = pd.DataFrame({"k=" + str(ap_list[0]): mse_result[0, 0]}, index=["GT-NMF"])
    df_mse.loc["GT-SNMF"] = mse_result[0, 1]
    df_mse.loc["NMF-SNMF"] = mse_result[0, 2]

    for ap_i, ap_s in enumerate(ap_list[1:]):
        df_mse["k=" + str(ap_s)] = mse_result[ap_i + 1, :]

    df_mse.to_csv(w_path + "MSE(GT_NMF_SNMF).csv")
    ipf.checkerboard_table(df_mse, w_path)

