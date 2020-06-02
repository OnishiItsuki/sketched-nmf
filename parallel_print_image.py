import functionfile as ff
import image_processing_function as ipf
import numpy as np
import pandas as pd
import os
import sys
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_images(c_mode, program_code, program_num, v, w, h, w_s, h_s,  r, ap, nmf_error, snmf_error, ite, wh_seed,
                 real_or_not, snmf_only=False, im_var=None, im_hol=None, use_data=None, image_num_list=None, c_bar_max=None):
    if c_mode == 0:
        c_method = "MU"
    elif c_mode == 1:
        c_method = "HALS"
    elif c_mode == 2:
        c_method = "FGD"
    elif c_mode == 3:
        c_method = "GCD"

    if real_or_not:
        nmf_v = np.dot(w, h)
        snmf_v = np.dot(w_s, h_s)

        # error_map ----------------------------------------------------------------------------------------------------
        plt.rcParams['font.size'] = 14
        error_map = np.zeros([im_var, im_hol, len(image_num_list), 3])

        for im_i, im_num in enumerate(image_num_list):
            write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/{}/Emap_wh_seed={}/" \
                .format(use_data, program_num, c_method, "error_map", wh_seed)

            error_map[:, :, im_i, 0] = np.reshape(v[:, im_num] - nmf_v[:, im_num], [im_var, im_hol])
            error_map[:, :, im_i, 1] = np.reshape(v[:, im_num] - snmf_v[:, im_num], [im_var, im_hol])
            error_map[:, :, im_i, 2] = np.reshape(nmf_v[:, im_num] - snmf_v[:, im_num], [im_var, im_hol])

        for data_num in range(3):
            ipf.save_error_map(error_map[:, :, :, data_num], c_bar_max, data_num, r, ap, write_path,  True)
            ipf.save_error_map(error_map[:, :, :, data_num], c_bar_max, data_num, r, ap, write_path, False)

        # calcurate_v_then_save ----------------------------------------------------------------------------------------
        nmf_v_list = np.zeros([im_var, im_hol, len(image_num_list)])
        snmf_v_list = np.zeros([im_var, im_hol, len(image_num_list)])

        write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/{}/wh_seed={}/" \
            .format(use_data, program_num, c_method, "pictures", wh_seed)
        os.makedirs(write_path, exist_ok=True)

        for im_i, im_num in enumerate(image_num_list):
            nmf_v_list[:, :, im_i] = np.reshape(nmf_v[:, im_num], [im_var, im_hol])
            snmf_v_list[:, :, im_i] = np.reshape(snmf_v[:, im_num], [im_var, im_hol])
        # print(nmf_v_list.shape)

        ipf.save_four_block_sample(nmf_v_list, write_path, "nmf_r={},ap={}".format(r, ap), "Existing method")
        ipf.save_four_block_sample(snmf_v_list, write_path, "snmf_r={},ap={}".format(r, ap), "Proposed method")

    # base result print  -----------------------------------------------------------------------------------------------
    if wh_seed == 1:
        if real_or_not:
            write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/r,k/r={}/k={}/graph/" \
                .format(use_data, program_num, c_method, r, ap)
        else:
            write_path = "/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/{}/{}_iteration{}/r,k/r={}/k={}/graph/" \
                .format(program_num, c_method, ite, r, ap)
        ipf.ite_cost_plot(nmf_error, snmf_error, ite, write_path, program_code)


if __name__ == "__main__":
    # Program_num = "10(for_MIRU)"
    Program_num = "11(for_MIRU_FGD_bag_fix)"
    # Program_num = "17(for_MIRU)"
    # Program_num = "18(for_MIRU_FGD_bag_fix)"

    # check!!!!!!!!!!!!!!!!!!!!!!!!
    C_mode = 2  # 0:MU NMF  1:HALS NMF  2:FGD  3:GCD
    mode = 2  # 2: V eval   3: W eval
    V_seed = 0

    if C_mode == 0:
        Color_bar_max = 70
        read_iteration = 5000
        # read_iteration = 3000
        C_method = "MU"
    elif C_mode == 1:
        Color_bar_max = 70
        read_iteration = 500
        # read_iteration = 400
        C_method = "HALS"
    elif C_mode == 2:
        Color_bar_max = 70
        # read_iteration = 500
        read_iteration = 1000
        C_method = "FGD"
    elif C_mode == 3:
        Color_bar_max = 70
        # read_iteration = 500
        read_iteration = 1000
        C_method = "GCD"

    if mode == 2:
        v_or_w = "v"
    elif mode == 3:
        v_or_w = "w"
    tmp = sys.argv

    r_size = int(tmp[1])
    approximate_size = int(tmp[2])
    # approximate_size = 10000
    WH_seed = int(tmp[3])  # base is 1
    test_flag = False

    if tmp[4] == "cbcl":
        Real_data = True
        Use_data = "CBCL/train"
        Image_num_list = [0, 500, 1000, 1500]
        mode = 2
        head = "realdata_"
        iteration = read_iteration
    elif tmp[4] == "yale":
        Real_data = True
        Use_data = "YaleFD/faces"
        Image_num_list = [0, 20, 40, 60]
        mode = 2
        head = "realdata_"
        iteration = read_iteration
    elif tmp[4] == "random":
        n = 100
        m = 10000
        V = Im_var = Im_hol = Image_num_list = 0
        Real_data = False
        Use_data = "random_matrix"
        V_seed = WH_seed - 1
        head = ""
        iteration = copy.deepcopy(read_iteration)
        read_iteration = 5000

    if tmp[5] == "test":
        Program_num = "test"
        read_iteration = 5
        iteration = read_iteration
        # iteration = 50
        test_flag = True

    print("start  r={}  k={}  WH_seed={}  {}_{}  {}  {}  >>>>>"
          .format(r_size, approximate_size, WH_seed, C_method, iteration, tmp[4], tmp[5]))

    # read V  ----------------------------------------------------------------------------------------------------------
    if Real_data:
        V, Im_var, Im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + Use_data)
        if V.shape[0] > V.shape[1]:
            m, n = V.shape
        else:
            n, m = V.shape

    # read NMF data  ---------------------------------------------------------------------------------------------------
    if test_flag | (not Real_data):
        read_path, read_file_name = ff.path_and_file_name(r_size, approximate_size, read_iteration, WH_seed, Use_data,
                                                          Program_num, C_method, "r,k", real_or_not=Real_data, v_or_w=v_or_w)
    else:
        read_path, read_file_name = ff.path_and_file_name(r_size, m, read_iteration, WH_seed, Use_data,
                                                          Program_num, C_method, "r,k", real_or_not=Real_data)
    if Real_data:
        NMF_error = 0
        W, H = ipf.read_wh(read_path + "matrix/", read_file_name, snmf=False)
    else:
        W, H = [0, 0]
    df = pd.read_csv("{}error/{}_error.csv".format(read_path, read_file_name))
    NMF_error = df["NMF error"]

    # read SNMF data  --------------------------------------------------------------------------------------------------
    read_path, read_file_name = ff.path_and_file_name(r_size, approximate_size, read_iteration, WH_seed, Use_data,
                                                      Program_num, C_method, "r,k", real_or_not=Real_data, v_or_w=v_or_w)
    if Real_data:
        SNMF_error = 0
        W_s, H_s = ipf.read_wh(read_path + "matrix/", read_file_name, snmf=True)
    else:
        W_s, H_s = [0, 0]
    df = pd.read_csv("{}error/{}_error.csv".format(read_path, read_file_name))
    SNMF_error = df["SNMF error"]

    Program_code = head + ff.program_name(mode, C_mode, n, m, r_size, approximate_size, read_iteration, V_seed, WH_seed, Program_num)
    print_images(C_mode, Program_code, Program_num, V, W, H, W_s, H_s, r_size, approximate_size, NMF_error, SNMF_error,
                 iteration, WH_seed, Real_data, im_hol=Im_hol, im_var=Im_var, use_data=Use_data,
                 image_num_list=Image_num_list, c_bar_max=Color_bar_max)
