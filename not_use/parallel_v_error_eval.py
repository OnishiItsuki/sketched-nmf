from joblib import Parallel, delayed
import functionfile as ff
import numpy as np
from operator import itemgetter

def rapid_v_error_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, nmfqp=False):
    nmf_error = np.zeros(iteration)
    snmf_error = np.zeros(iteration)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v, approximate_size, 0, t=True)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

    w_s = np.zeros([n, r, iteration + 1])
    w_s[:, :, 0] = ff.generate_w(n, r, seeds[0], c_mode=c_mode)

    if nmfqp:
        w = np.zeros([n, r, iteration + 1])
        w[:, :, 0] = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    else:
        w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)

    print("\n\n\n------------------   NMF   -------------------")
    # if nmfqp:
    #     print("NMF matrix H is calculated by QP.")
    #     for i in range(iteration):
    #         w[:, :, i + 1], h, nmf_error[i] = calculate(v, w[:, :, i], h, c_mode, i)
    #     tmp = Parallel(n_jobs=-1, verbose=3)([delayed(ff.calculate_h)(v, w[:, :, i], qp_num=i) for i in range(iteration)])
    # else:
    #     for i in range(iteration):
    #         w, h, nmf_error[i] = calculate(v, w, h, c_mode, i)

    print("\n\n-------------   Sketching NMF   --------------")
    for i in range(0, iteration):
        w_s[:, :, i + 1], h_s, snmf_error[i] = calculate(v_s, w_s[:, :, i], h_s, c_mode, i)
    tmp = Parallel(n_jobs=3, verbose=3)([delayed(ff.calculate_h)(v, w_s[:, :, i + 1], qp_num=i) for i in range(iteration)])
    # h_os = ff.calculate_h(v, w_s, print_interim=True)

    tmp.sort(key=itemgetter(1))

    return nmf_error, snmf_error, w[:, :, iteration], h, w_s[:, :, iteration], 0

def calculate(v, w, h, c_mode, i, qp_opt=False):
    w, h = ff.update(v, w, h, c_mode)
    if qp_opt:
        h_qp = ff.calculate_h(v, w, print_interim=True)
        frobenius_norm = np.linalg.norm(v - np.dot(w, h_qp)) ** 2
    else:
        frobenius_norm = np.linalg.norm(v - np.dot(w, h)) ** 2

    if (i == 0) | (i % 100 == 99):
        print(str(i + 1) + " times update  error: " + str(frobenius_norm))

    return w, h, frobenius_norm

