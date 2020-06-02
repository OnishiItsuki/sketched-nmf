import functionfile as ff
import image_processing_function as ipf
import numpy as np
import pandas as pd
import sys
import os


def bic(v, r, iteration, seed):
    n, m = v.shape
    seeds = ff.two_seeds(seed)
    w = ff.generate_w(n, r, seeds[0], c_mode=0)
    h = ff.generate_h(r, m, seeds[1], c_mode=0)
    error = np.zeros(iteration)

    for i in range(iteration):
        w, h = ff.update(v, w, h, 0)
        error[i] = np.linalg.norm(v - np.dot(w, h)) ** 2
        if i % 1000 == 999:
            print("BIC  NMF ( r={} ) : {} times update".format(r, i + 1))
    v_reconstructed = np.dot(w, h)

    a = (n + m) / (n * m)
    d_eu = np.linalg.norm(v - v_reconstructed) ** 2
    return np.log(d_eu) + r * a * np.log(1 / a), error


def bcv(v, r, iteration, seed, row_sep_num, column_sep_num, error_print):
    n, m = v.shape
    n_sep_size = int(n / row_sep_num)
    m_sep_size = int(m / column_sep_num)
    counter = 1
    bic_tmp = 0
    seeds = ff.two_seeds(seed)
    error = np.zeros(iteration)

    if n - n_sep_size < r | m - m_sep_size < r:
        print("Error: row or column size of D is bigger than R", file=sys.stderr)
        sys.exit(1)

    for r_sep in range(row_sep_num):
        for c_sep in range(column_sep_num):  # for folded row and column
            row_s = n_sep_size * r_sep
            column_s = m_sep_size * c_sep

            if r_sep != row_sep_num - 1:
                row_e = n_sep_size * (r_sep + 1)
            else:
                row_e = n
            if c_sep != column_sep_num - 1:
                column_e = m_sep_size * (c_sep + 1)
            else:
                column_e = m

            # set Matrices ABCD  ------------------------------------------------------------------------------------
            a = v[row_s:row_e, column_s:column_e]
            b = np.concatenate([v[row_s:row_e, :column_s], v[row_s:row_e, column_e:]], 1)
            c = np.concatenate([v[:row_s, column_s:column_e], v[row_e:, column_s:column_e]], 0)

            d_1 = np.concatenate([v[:row_s, :column_s], v[:row_s, column_e:]], 1)
            d_2 = np.concatenate([v[row_e:, :column_s], v[row_e:, column_e:]], 1)
            d = np.concatenate([d_1, d_2], 0)

            # fit NMF to D  --------------------------------------------------------------------------------
            w_d = ff.generate_w(d.shape[0], r, seeds[0], c_mode=0)
            h_d = ff.generate_h(r, d.shape[1], seeds[1], c_mode=0)

            for i in range(iteration):
                w_d, h_d = ff.update(d, w_d, h_d, 0)
                if error_print & (row_e == n) & (column_e == m):
                    error[i] = np.linalg.norm(d - np.dot(w_d, h_d)) ** 2
                if i % 1000 == 999:
                    print("BCV  NMF ( r={}  {} / {}) : {} times update".format(r, counter, row_sep_num * column_sep_num, i + 1))

            # calculate BCV
            a_r = np.dot(np.dot(b, np.linalg.pinv(h_d)), np.dot(np.linalg.pinv(w_d), c))
            bic_tmp += np.linalg.norm(a - a_r) ** 2
            counter += 1
    return bic_tmp, error


if __name__ == "__main__":
    ite = 1500
    row_separate_num = 5
    column_separate_num = 5
    write_dir = "r_choice"

    D_error_print = True

    tmp = sys.argv
    R = int(tmp[1])
    SEED = int(tmp[2])
    if tmp[4] == "cbcl":
        Real_or_not = True
        Use_data = "CBCL/train"
    elif tmp[4] == "yale":
        Real_or_not = True
        Use_data = "YaleFD/faces"
        head = "realdata_"
    elif tmp[4] == "random":
        Real_or_not = False

    if tmp[5] == "test":
        write_dir = "test_r_choice"
        ite = 5
        row_separate_num = 2
        column_separate_num = 2
        test_flag = True

    write_path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}".format(Use_data, write_dir)
    V, Im_var, Im_hol = ipf.read_pgm("/home/ionishi/mnt/workspace/sketchingNMF/face_data/" + Use_data)
    if V.shape[0] > V.shape[1]:
        V = V.T

    print("start  {} / {}  r={}  seed={}  ---{}  >>>>>>>>>".format(tmp[6], tmp[7], R, SEED, tmp[3]))

    if tmp[3] == "bic":
        os.makedirs(write_path + "/BIC", exist_ok=True)
        os.makedirs(write_path + "/LS", exist_ok=True)

        result, V_error = bic(V, R, ite, SEED)
        np.savetxt("{}/BIC/r{}_seed{}.csv".format(write_path, R, SEED), result.reshape(1,), delimiter=",")
        np.savetxt("{}/LS/r{}_seed{}.csv".format(write_path, R, SEED), V_error, delimiter=",")

    elif tmp[3] == "bcv":
        os.makedirs("{}/BCV_sep({},{})".format(write_path, row_separate_num, column_separate_num), exist_ok=True)
        os.makedirs("{}/D_LS_sep({},{})".format(write_path, row_separate_num, column_separate_num), exist_ok=True)

        result, D_error = bcv(V, R, ite, SEED, row_separate_num, column_separate_num, D_error_print)
        np.savetxt("{}/BCV_sep({},{})/r{}_seed{}.csv"
                   .format(write_path, row_separate_num, column_separate_num, R, SEED), result.reshape(1,), delimiter=",")
        if D_error_print:
            np.savetxt("{}/D_LS_sep({},{})/r{}_seed{}.csv"
                       .format(write_path, row_separate_num, column_separate_num, R, SEED), D_error, delimiter=",")

    print("\nfinish   {} / {}   r={}  seed={}  ---{}  >>>>>\n".format(tmp[6], tmp[7], R, SEED, tmp[3]))
