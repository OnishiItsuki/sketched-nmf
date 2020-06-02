import functionfile as ff
import image_processing_function as ipf
import numpy as np
import os

# program_num = "7(change seed no QP)"
# program_num = "8(change seed NMF QP)"
program_num = "9(for_thesis)"


use_data = "CBCL/train"
# use_data = "YaleFD/faces"

c_method = "MU"
# c_method = "HALS"

w_program_num = program_num
# w_program_num = "im_test"

real_or_not = True
space = False

# =================check this when error printed ================
if "MU" in c_method:
    ite = 3000
elif "HALS" in c_method:
    ite = 400

mode = 1  # 0:save_original_image  1:save_w_image  2:error_map  3:calculate_v_then_save
c_bar_max = 70
seed_list = range(1, 11)

if "CBCL" in use_data:
    plot_r_list = [50]
    # plot_ap_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2429]
    plot_ap_list = [2429]
    image_num_list = [0, 500, 1000, 1500]
    if "test" in w_program_num:
        plot_ap_list = [250]
        seed_list = [1]
elif "YaleFD" in use_data:
    plot_r_list = [25]
    plot_ap_list = [10000]
    image_num_list = [0, 20, 40, 60]
    if "test" in w_program_num:
        plot_r_list = [15]
        plot_ap_list = [20]
        seed_list = [1]

if mode == 0:
    read_path = "../mnt/workspace/sketchingNMF/face_data/" + use_data
    write_path = "../mnt/workspace/sketchingNMF/" + use_data + "/sample/"

    os.makedirs(write_path, exist_ok=True)
    for im_num in image_num_list:
        ipf.save_original_image(read_path, write_path, str(image_num_list), image_num_list)

elif mode == 1:
    original_v, im_var, im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data)
    image_size = [im_var, im_hol]

    if "CBCL" in use_data:
        read_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/r,k/r=50/k=2429/matrix".format(use_data, program_num, c_method)
        read_file_name = "w_realdata_9(for_thesis)_V_MU_[n361,m2429,r50,as2429,ite3000,seed0,1]"
    elif "Yale" in use_data:
        read_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/r,k/r=25/k=77760/matrix".format(use_data, program_num, c_method)
        read_file_name = "w_realdata_9(for_thesis)_V_MU_[n165,m77760,r25,as77760,ite3000,seed0,1]"
    write_path = "/home/ionishi/mnt/workspace/sketchingNMF/" + use_data + "/" + program_num + "/images"
    for im_num in range(8):
        ipf.save_w_image(read_path, read_file_name, write_path, image_size, im_num)

elif mode == 2:
    print("start {}  {} statistics -> wh_seed {}".format(use_data, c_method, seed_list))

    for wh_seed in seed_list:
        original_v, im_var, im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data)
        error_map = np.zeros([im_var, im_hol, len(image_num_list), 3])

        for r in plot_r_list:
            for ap in plot_ap_list:
                for im_i, im_num in enumerate(image_num_list):
                    read_path, read_file_name = ff.path_and_file_name(r, ap, ite, wh_seed, use_data, program_num, c_method,
                                                                      "r,k", real_or_not=real_or_not, directory="matrix", space=space)
                    write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/{}/Emap_wh_seed={}/" \
                        .format(use_data, w_program_num, c_method, "error_map", wh_seed)
                    os.makedirs(write_path, exist_ok=True)
                    Emap_nmf_v = ipf.calculate_v(read_path, read_file_name, snmf=False)
                    Emap_snmf_v = ipf.calculate_v(read_path, read_file_name, snmf=True)

                    error_map[:, :, im_i, 0] = np.reshape(original_v[:, im_num] - Emap_nmf_v[:, im_num], [im_var, im_hol])
                    error_map[:, :, im_i, 1] = np.reshape(original_v[:, im_num] - Emap_snmf_v[:, im_num], [im_var, im_hol])
                    error_map[:, :, im_i, 2] = np.reshape(Emap_nmf_v[:, im_num] - Emap_snmf_v[:, im_num], [im_var, im_hol])

                for data_num in range(3):
                    ipf.save_error_map(error_map[:, :, :, data_num], c_bar_max, data_num, r, ap, write_path, True)
                    ipf.save_error_map(error_map[:, :, :, data_num], c_bar_max, data_num, r, ap, write_path, False)

        print("finish error map  wh_seed {} / {}".format(wh_seed, len(seed_list)))
elif mode == 3:
        original_v, im_var, im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + use_data)

        for r in plot_r_list:
            for ap in plot_ap_list:
                read_path, read_file_name = ff.path_and_file_name(r, ap, ite, seed_list[0], use_data, program_num, c_method,
                                                                  "r,k", real_or_not=real_or_not, directory="matrix", space=space)
                write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/images/".format(use_data, w_program_num)
                os.makedirs(write_path, exist_ok=True)

                nmf_v_tmp = ipf.calculate_v(read_path, read_file_name, snmf=False)
                snmf_v_tmp = ipf.calculate_v(read_path, read_file_name, snmf=True)

                ipf.save_sample(nmf_v_tmp, im_var, im_hol, write_path, "nmf_r={}_ap={}".format(r, ap))
        print("finish wh_seed {} / {}".format(seed_list[0], len(seed_list)))
