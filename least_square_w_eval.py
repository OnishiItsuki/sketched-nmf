import numpy as np
import functionfile as ff


def least_square_w_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, o_w=0):
    nmf_error = np.zeros(iteration)
    snmf_error = np.zeros(iteration)
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v, approximate_size, seeds[0] + 1, t=True)
    w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

    if np.all(o_w == 0):
        print("\n\n\n------------------   NMF   -------------------")
        for i in range(0, iteration):
            w, h = ff.update(v, w, h, r, c_mode)
            w_s, h_s, nmf_error[i] = calculate(v, w, w_s, h_s, c_mode, i)
        return nmf_error, snmf_error, w, h, w_s, h_s

    else:
        print("\n\n\n------------------   NMF   -------------------")
        for i in range(0, iteration):
            w, h, nmf_error[i] = calculate(v, o_w, w, h, c_mode, i)

        print("\n\n-------------   Sketching NMF   --------------")
        for i in range(0, iteration):
            w_s, h_s, snmf_error[i] = calculate(v_s, o_w, w_s, h_s, c_mode, i)
        h_os = ff.calculate_h(v, w_s, print_interim=True)

        return nmf_error, snmf_error, w, h, w_s, h_os


def calculate(v, o_w, w, h, c_mode, i):
    w, h = ff.update(v, w, h, c_mode)
    d = np.linalg.lstsq(w, o_w)[0]
    frobenius_norm = np.linalg.norm(w - np.dot(w, d)) ** 2

    if (i == 0) | (i % 100 == 99):
        print(str(i + 1) + " times update  error: " + str(frobenius_norm))

    return w, h, frobenius_norm


def parallel_least_square_w_eval(n, m, r, approximate_size, v, o_w, iteration, wh_seed, c_mode, column_sketching):
    seeds = ff.two_seeds(wh_seed)
    nmf_error = np.zeros(iteration)
    snmf_error = np.zeros(iteration)
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v, approximate_size, seeds[0] + 1, right_product=column_sketching)
    w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

    for i in range(0, iteration):
        w, h, nmf_error[i] = calculate(v, o_w, w, h, c_mode, i)

    for i in range(0, iteration):
        w_s, h_s, snmf_error[i] = calculate(v_s, o_w, w_s, h_s, c_mode, i)
    h_os = ff.calculate_h(v, w_s)

    return nmf_error, snmf_error, w, h, w_s, h_os
