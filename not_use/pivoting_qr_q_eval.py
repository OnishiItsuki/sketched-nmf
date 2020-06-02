import numpy as np
from scipy import linalg
import functionfile as ff


def pivoting_qr_q_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, o_w=0):
    print("\n\n\n------------------   NMF   -------------------")

    nmf_error = np.zeros(iteration)
    snmf_error = np.zeros(iteration)
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v, approximate_size, 0, t=True)
    w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

    if np.all(o_w == 0):
        for i in range(0, iteration):
            w, h = ff.update(v, w, h, r, c_mode)
            qr_q, qr_r, p = linalg.qr(w, pivoting=True)

            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
            qr_q_s, qr_r_s, p_s = linalg.qr(w_s, pivoting=True)

            frobenius_norm = np.linalg.norm(np.dot(qr_q_s, qr_r_s) - np.dot(qr_q, qr_r)) ** 2
            nmf_error[i] = frobenius_norm

            if (i == 0) | (i % 100 == 99):
                print(str(i + 1) + " times update  error: " + str(frobenius_norm))
        return nmf_error, snmf_error, w, h, w_s, h_s

    else:
        original_q, original_r, original_p = linalg.qr(o_w, pivoting=True)
        o_w = np.dot(original_q, original_r)

        # NMF------------------------------------
        print("\n\n\n------------------   NMF   -------------------")

        for i in range(0, iteration):
            nmf_error[i] = calculate(v, o_w, w, h, c_mode, i)

        print("\n\n-------------   Sketching NMF   --------------")
        for i in range(0, iteration):
            snmf_error[i] = calculate(v_s, o_w, w_s, h_s, c_mode, i)
        h_os = ff.calculate_h(v, w_s, True)

        return nmf_error, snmf_error, w, h, w_s, h_os


def calculate(v, o_w, w, h, c_mode, i):
    w, h = ff.update(v, w, h, c_mode)
    qr_q, qr_r, p = linalg.qr(w, pivoting=True)
    frobenius_norm = np.linalg.norm(o_w - np.dot(qr_q, qr_r)) ** 2

    if (i == 0) | (i % 100 == 99):
        print(str(i + 1) + " times update  error: " + str(frobenius_norm))

    return w, h, frobenius_norm
