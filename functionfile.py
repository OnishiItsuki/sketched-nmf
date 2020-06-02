import scipy.sparse as sp
import numpy as np
import pandas as pd
import cvxopt
from cvxopt import matrix
import sys


def uniform_sampling(a, approxi_size, seed, right_product=True):
    n, m = a.shape
    np.random.seed(seed)

    # C = AS (return n * s matrix) s:approxi_size
    if right_product:
        if approxi_size > a.shape[1]:  # ap < m
            print("ff.uniform_sampling Error: approximation_size [{}] is not smaller than size of row [{}]".format(approxi_size, a.shape[1]), file=sys.stderr)
            sys.exit(1)

        id_ = np.arange(m)
        id = np.random.choice(id_, size=approxi_size, replace=False)
        c = a[:, id]

    # C = SA (return s * m matrix) s:approxi_size
    else:
        if approxi_size > a.shape[0]:  # ap < n
            print("ff.uniform_sampling Error: approximation_size [{}] is not smaller than size of column [{}]".format(approxi_size, a.shape[1]), file=sys.stderr)
            sys.exit(1)

        id_ = np.arange(n)
        id = np.random.choice(id_, size=approxi_size, replace=False)
        c = a[id, :]
    return c


def generate_wh(row_size, column_size, r_size, seed):
    if (r_size >= row_size) & (r_size >= column_size):  # r < min(n,m)
        print("ff.generate_wh Error: r_size [{}] is not smaller than size of row [{}] or columns [{}]".format(row_size, row_size, column_size), file=sys.stderr)
        sys.exit(1)

    seeds = two_seeds(seed)
    # print("seed of original W=" + str(seeds[0]) + "  seed of original H=" + str(seeds[1]))
    w = generate_w(row_size, r_size, seeds[0])
    h = generate_h(r_size, column_size, seeds[1])
    return np.dot(w, h), w, h


def generate_w(row_size, r_size, seed, c_mode=0):
    np.random.seed(seed)
    w = np.random.rand(row_size, r_size)
    if np.linalg.matrix_rank(w) != r_size:  # check rank
        print("ff.generate_w Error: rank of W does not match R size.  rank=" + str(np.linalg.matrix_rank(w)) + "  R_size="
              + str(r_size), file=sys.stderr)
        sys.exit(1)
    if c_mode == 1:
        w = normalize_column_vectors(w)
    return w


def generate_h(r_size, column_size, seed, c_mode=0):
    np.random.seed(seed)
    h = np.random.rand(r_size, column_size)
    if c_mode == 1:
        h = normalize_column_vectors(h.T).T
    return h


def update(v, w, h, c_mode):
    if c_mode == 0:   # MU update
        h = h * np.dot(w.T, v) / np.dot(w.T, np.dot(w, h))
        w = w * np.dot(v, h.T) / np.dot(np.dot(w, h), h.T)
        return w, h

    elif c_mode == 1:  # HALS update
        b = h.T
        # update B
        for j in range(0, b.shape[1]):  # every column calculate
            b[:, j] = np.fmax(10e-16, b[:, j] + np.dot(v.T, w)[:, j] - np.dot(b, np.dot(w.T, w)[:, j]))
        # update W
        for j in range(0, w.shape[1]):  # every column calculate
            w[:, j] = np.fmax(10e-16, w[:, j] * np.dot(b.T, b)[j, j] + np.dot(v, b)[:, j] - np.dot(w, np.dot(b.T, b)[:, j]))
            w[:, j] /= np.linalg.norm(w[:, j], ord=2)
        return w, b.T
    elif c_mode == 3:  # GCD update
        upsilon = 10e-3
        # H update -----------------------------------------------------------------------------------------------------
        p_wv = np.dot(w.T, v)
        p_ww = np.dot(w.T, w)
        g_h = np.dot(p_ww, h) - p_wv

        h = gcd_inner(g_h, p_ww, h, upsilon)

        # W update -----------------------------------------------------------------------------------------------------
        p_vh = np.dot(v, h.T)
        p_hh = np.dot(h, h.T)
        g_w = np.dot(w, p_hh) - p_vh

        w = gcd_inner(g_w.T, p_hh, w.T, upsilon).T

        return w, h


def fgd_update(v, w, h, theta_w, theta_h):
    stop = 10e-5
    # update h ---------------------------------------------------------------------------------------------------------
    nabra_h = h * np.dot(w.T, v) / np.dot(w.T, np.dot(w, h)) - h
    while True:
        tmp = theta_h + np.sum(np.dot(w, nabra_h) * (v - np.dot(w, h + theta_h * nabra_h))) / np.linalg.norm(np.dot(w, nabra_h)) ** 2
        if np.abs(tmp - theta_h) < stop:
            theta_h = min(tmp, 0.01 + 0.99 * (h / -1 * nabra_h).max())
            h += theta_h * nabra_h
            break
        else:
            theta_h = tmp

    # update w ---------------------------------------------------------------------------------------------------------
    nabra_w = w * np.dot(v, h.T) / np.dot(np.dot(w, h), h.T) - w
    while True:
        tmp = theta_w + np.sum(np.dot(nabra_w, h) * (v - np.dot(w + theta_w * nabra_w, h))) / np.linalg.norm(np.dot(nabra_w, h)) ** 2
        if np.abs(tmp - theta_w) < stop:
            theta_w = min(tmp, 0.01 + 0.99 * (w / -1 * nabra_w).max())
            w += theta_w * nabra_w
            break
        else:
            theta_w = tmp

    return w, h, theta_w, theta_h


def gcd_inner(g_h, p_ww, h, upsilon):
    p_wwrr = np.zeros(h.shape)  # r times r
    for r in range(p_wwrr.shape[0]):
        p_wwrr[r, :] = np.full(p_wwrr.shape[1], p_ww[r, r])  # all values of ith column vector are w[i, i] ** 2

    s_h = np.maximum(0, h - (g_h / p_wwrr)) - h  # 4
    d_h = - (g_h * s_h) - (p_wwrr * s_h * s_h / 2)  # 5
    p_init = 0  # 6
    q = np.zeros(h.shape[1])
    for i in range(h.shape[1]):
        q[i] = np.int(np.argmax(d_h[:, i]))
        if p_init < d_h[q[i], i]:
            p_init = d_h[q[i], i]

    # inner loop
    for i in range(h.shape[1]):
        while d_h[q[i], i] > upsilon * p_init:
            s_ = s_h[q[i], i]  # 7.1
            h[q[i], i] += s_  # 7.3
            g_h[:, i] += s_ * p_ww[:, q[i]]  # 7.4
            s_h[:, i] = np.maximum(0, h[:, i] - (g_h[:, i] / p_wwrr[:, i])) - h[:, i]  # 7.5
            d_h[:, i] = - (g_h[:, i] * s_h[:, i]) - (p_wwrr[:, i] * s_h[:, i] * s_h[:, i] / 2)  # 7.6
            q[i] = np.argmin(d_h[:, i])  # 7.7
    return h


def normalize_column_vectors(m):
    for i in range(0, m.shape[1]):
        m[:, i] = m[:, i] / np.linalg.norm(m[:, i], ord=2)
    return m


def calculate_h(v, w, qp_num=None,  print_interim=False):
    cvxopt.solvers.options['show_progress'] = False
    result = np.empty((w.shape[1], v.shape[1]))
    p = matrix(2 * np.dot(w.T, w))
    g = matrix(-1 * np.identity(w.shape[1]))
    h = matrix(np.zeros(w.shape[1]).T)

    for i in range(0, v.shape[1]):  # vの列数繰り返す
        q = matrix(-2 * np.dot(v.T[i, :], w))
        sol = cvxopt.solvers.qp(p, q, g, h)
        result[:, i] = np.array(sol["x"]).T
        if print_interim & ((i == 0) | (i % 100 == 99)):
            print("\r{0} times qp".format(i+1), end="")

    if qp_num is None:
        return result
    else:
        return result, qp_num


def two_seeds(seed):
    np.random.seed(seed)
    tmp = np.random.randint(0, 1000, 2)
    return tmp


def program_name(mode, c_mode, n, m, r, approxi_size, iterator, v_seed, wh_seed,  program_code, convergence_D_mode=0, CDC=0):
    if (mode < 0) & (mode > 3):
        print("ff,program_name Error: this mode does not exit", file=sys.stderr)
        sys.exit(1)

    if convergence_D_mode == 1:
        cdm = "absol_cdc"
    elif convergence_D_mode == 2:
        cdm = "diff_cdc"

    if mode == 0:
        name = "time"
    elif mode == 1:
        name = "Q"
    elif mode == 2:
        name = "V"
    elif mode == 3:
        name = "ls_W"

    if c_mode == 0:
        c_name = "MU"
    elif c_mode == 1:
        c_name = "HALS"
    elif c_mode == 2:
        c_name = "FGD"
    elif c_mode == 3:
        c_name = "GCD"

    p_c = str(program_code)
    n = str(n)
    m = str(m)
    r = str(r)
    ite = str(iterator)
    a_s = str(approxi_size)
    vs = str(v_seed)
    whs = str(wh_seed)
    CDC = str(CDC)

    if convergence_D_mode == 0:
        return p_c + "_" + name + "_" + c_name + "_[n" + n + ",m" + m + ",r" + r + ",as" + a_s + ",ite" + ite + \
               ",seed" + vs + "," + whs + "]"
    else:
        return p_c + "_" + name + "_" + c_name + "_" + cdm + CDC + "_[n" + n + ",m" + m + ",r" + r + ",as" + a_s + \
               ",ite" + ite + ",seed" + vs + "," + whs + "]"


def ct_save_program_name(mode, c_mode, iteration, program_code, convergence_D_mode, CDC):
    if (mode < 0) & (mode > 3):
        print("ff.ct_save_program_nameError: this mode does not exit", file=sys.stderr)
        sys.exit(1)

    if convergence_D_mode == 1:
        cdm = "absol_cdc_"
    elif convergence_D_mode == 2:
        cdm = "diff_cdc_"

    if mode == 0:
        name = "time"
    elif mode == 1:
        name = "Q"
    elif mode == 2:
        name = "V"
    elif mode == 3:
        name = "ls_W"

    if c_mode == 0:
        c_name = "MU"
    elif c_mode == 1:
        c_name = "HALS"
    elif c_mode == 2:
        c_name = "FGD"
    elif c_mode == 3:
        c_name = "GCD"

    p_c = str(program_code)
    ite = str(iteration)
    CDC = str(CDC)

    return p_c + "_" + c_name + "_" + cdm + CDC + "ite_" + ite


def print_interim_progress(error, i, mode):
    if mode == 0:
        print("\r" + str(i + 1) + " times update", end="")
    elif (mode >= 1) & (mode <= 3):
        print(str(i + 1) + " times update  FN_error: " + str(error))


def path_and_file_name(r, ap, ite, wh_seed, use_data, program_num, c_method, pic_or_data, n=0, m=0, v_or_w="v", real_or_not=True, directory=None, space=False):
    if n == 0 & m == 0:
        if "CBCL" in use_data:
            n = 361
            m = 2429
        elif "YaleFD" in use_data:
            m = 77760
            n = 165
            # n = 77760
            # m = 165
        elif "random" in use_data:
            n = 100
            m = 10000
    elif n == 0 | m == 0:
        print("ff.path_and_file_name Error: n or m is undefined", file=sys.stderr)
        sys.exit(1)

    if "random" in use_data:
        ra_or_re = ""
        se0 = wh_seed - 1
        path_c_method = c_method + "_iteration" + str(ite)
    else:
        ra_or_re = "realdata_"
        se0 = "0"
        path_c_method = c_method

    if directory == "time":
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cal_method = "time"
        # cal_method = "V"
    else:
        if v_or_w == "w":
            cal_method = "ls_W"
        else:
            cal_method = "V"

    # This code has mistake. You get error if V_evaluate mode did not run.
    if space:
        file_name = "{}{}_{}_{}_[n{}, m{}, r{}, as{}, ite{},seed {}, {}]".format(ra_or_re, program_num, cal_method, c_method, n, m, r, ap, ite, se0, wh_seed)
    else:
        file_name = "{}{}_{}_{}_[n{},m{},r{},as{},ite{},seed{},{}]".format(ra_or_re, program_num, cal_method, c_method, n, m, r, ap, ite, se0, wh_seed)

    if directory is None:
        path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/{}/r={}/k={}/".format(use_data, program_num, path_c_method, pic_or_data, r, ap)
    else:
        path = "/home/ionishi/mnt/workspace/sketchingNMF/{}/{}/{}/{}/r={}/k={}/{}/".format(use_data, program_num, path_c_method, pic_or_data, r, ap, directory)
        # This code has mistake. You get error if V_evaluate mode did not run.
    return path, file_name


def parallel_make_list(bash_input):
    r_list_size = int(bash_input[1])
    ap_list_size = int(bash_input[2])
    seed_list_size = int(bash_input[3])

    r_list = []
    ap_list = []
    seed_list = []
    count = 4
    for i in range(r_list_size):
        r_list.append(bash_input[count])
        count += 1
    for i in range(ap_list_size):
        ap_list.append(bash_input[count])
        count += 1
    for i in range(seed_list_size):
        seed_list.append(bash_input[count])
        count += 1
    return r_list, r_list_size, ap_list, ap_list_size, seed_list, seed_list_size


def read_cost_func_error(r_size, approximate_size, iteration, wh_seed, use_data, program_num, c_method, test_flag, n=0, m=0, v_or_w="v"):
    real_or_not = True
    if m == 0:
        if "CBCL" in use_data:
            m = 2429
        elif "YaleFD" in use_data:
            m = 77760
    if "random" in use_data:
        real_or_not = False

    # read NMF data  ---------------------------------------------------------------------------------------------------
    if test_flag | (not real_or_not):
        read_path, read_file_name = path_and_file_name(r_size, approximate_size, iteration, wh_seed, use_data,
                                                          program_num, c_method, "r,k", n=n, m=m, real_or_not=real_or_not, v_or_w=v_or_w)
    else:
        read_path, read_file_name = path_and_file_name(r_size, m, iteration, wh_seed, use_data,
                                                          program_num, c_method, "r,k", n=n, m=m, real_or_not=real_or_not, v_or_w=v_or_w)
    df = pd.read_csv("{}error/{}_error.csv".format(read_path, read_file_name))
    nmf_error = df["NMF error"].values

    # read SNMF data  --------------------------------------------------------------------------------------------------
    read_path, read_file_name = path_and_file_name(r_size, approximate_size, iteration, wh_seed, use_data,
                                                      program_num, c_method, "r,k", n=n, m=m, real_or_not=real_or_not, v_or_w=v_or_w)
    df = pd.read_csv("{}error/{}_error.csv".format(read_path, read_file_name))
    snmf_error = df["SNMF error"].values

    return nmf_error, snmf_error


def read_time(r_size, approximate_size, iteration, wh_seed, use_data, program_num, c_method, test_flag, n=0, m=0):
    real_or_not = True
    if m == 0:
        if "CBCL" in use_data:
            m = 2429
        elif "YaleFD" in use_data:
            m = 77760
    if "random" in use_data:
        real_or_not = False

    # read NMF data  ---------------------------------------------------------------------------------------------------
    if test_flag | (not real_or_not):
        read_path, read_file_name = path_and_file_name(r_size, approximate_size, iteration, wh_seed, use_data,
                                                       program_num, c_method, "r,k", n=n, m=m, real_or_not=real_or_not, directory="time")
    else:
        read_path, read_file_name = path_and_file_name(r_size, m, iteration, wh_seed, use_data,
                                                          program_num, c_method, "r,k", n=n, m=m, real_or_not=real_or_not, directory="time")
    df = pd.read_csv("{}{}.csv".format(read_path, read_file_name))
    nmf_time = df["NMF time"].values

    # read SNMF data  --------------------------------------------------------------------------------------------------
    read_path, read_file_name = path_and_file_name(r_size, approximate_size, iteration, wh_seed, use_data,
                                                      program_num, c_method, "r,k", n=n, m=m, real_or_not=real_or_not, directory="time")
    df = pd.read_csv("{}{}.csv".format(read_path, read_file_name))
    snmf_time = df["SNMF time"].values

    return nmf_time, snmf_time
