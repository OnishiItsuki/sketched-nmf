import numpy as np
import functionfile as ff


def v_error_eval(n, m, r, approximate_size, v, iteration, seeds, c_mode, nmfqp, column_sketching):
    nmf_error = np.zeros(iteration)
    snmf_error = np.zeros(iteration)
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v, approximate_size, seeds[0] + 1, right_product=column_sketching)

    print("\n\n\n------------------   NMF   -------------------")
    if nmfqp:
        print("NMF matrix H is calculated by QP.")
        for i in range(0, iteration):
            w, h = ff.update(v, w, h, c_mode)
            h_qp = ff.calculate_h(v, w, print_interim=True)
            nmf_error[i] = np.linalg.norm(v - np.dot(w, h_qp)) ** 2
            if (i == 0) | (i % 100 == 99):
                print(str(i + 1) + " times update  error: " + str(nmf_error[i]))
        h = h_qp

    else:
        for i in range(0, iteration):
            w, h = ff.update(v, w, h, c_mode)
            nmf_error[i] = np.linalg.norm(v - np.dot(w, h)) ** 2
            if (i == 0) | (i % 100 == 99):
                print(str(i + 1) + " times update  error: " + str(nmf_error[i]))

    print("\n\n-------------   Sketching NMF   --------------")
    if column_sketching:
        w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
        h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)
        for i in range(0, iteration):
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
            h_os = ff.calculate_h(v, w_s, print_interim=True)
            snmf_error[i] = np.linalg.norm(v - np.dot(w_s, h_os)) ** 2
            if (i == 0) | (i % 100 == 99):
                print(str(i + 1) + " times update  error: " + str(snmf_error[i]))
        h_s = h_os

    else:
        w_s = ff.generate_w(approximate_size, r, seeds[0], c_mode=c_mode)
        h_s = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
        for i in range(0, iteration):
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
            w_os = ff.calculate_h(v, h_s, print_interim=True)
            snmf_error[i] = np.linalg.norm(v - np.dot(w_os, h_s)) ** 2
            if (i == 0) | (i % 100 == 99):
                print(str(i + 1) + " times update  error: " + str(snmf_error[i]))
        w_s = w_os

    return nmf_error, snmf_error, w, h, w_s, h_s


def parallel_v_error_eval(r, approximate_size, v, iteration, wh_seed, c_mode, nmfqp, t_flag,  snmf_only=False):
    theta_start = 5
    n, m = v.shape
    seeds = ff.two_seeds(wh_seed)
    nmf_error = np.zeros(iteration)
    snmf_error = np.zeros(iteration)
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
    v_s = ff.uniform_sampling(v, approximate_size, seeds[0] + 1)

    # NMF  -------------------------------------------------------------------------------------------------------------
    theta_w = theta_h = theta_start
    if not snmf_only:
        if nmfqp:
            print("NMF matrix H is calculated by QP.")
            for i in range(0, iteration):
                if c_mode != 2:
                    w, h = ff.update(v, w, h, c_mode)
                else:
                    w, h, theta_w, theta_h = ff.fgd_update(v, w, h, theta_w, theta_h)
                h_qp = ff.calculate_h(v, w, print_interim=False)
                nmf_error[i] = np.linalg.norm(v - np.dot(w, h_qp)) ** 2
                if (i == 0) | (i % 100 == 99):
                    print("NMF ( r=" + str(r) + "  k=" + str(approximate_size) + " ) : " + str(i + 1) + " times update")
            h = h_qp

        else:
            for i in range(0, iteration):
                if c_mode != 2:
                    w, h = ff.update(v, w, h, c_mode)
                else:
                    w, h, theta_w, theta_h = ff.fgd_update(v, w, h, theta_w, theta_h)
                nmf_error[i] = np.linalg.norm(v - np.dot(w, h)) ** 2
                if (i == 0) | (i % 2000 == 1999):
                    print("NMF ( r=" + str(r) + "  k=" + str(approximate_size) + " ) : " + str(i + 1) + " times update")

    # SNMF  ------------------------------------------------------------------------------------------------------------
    w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)
    for i in range(0, iteration):
        if c_mode != 2:
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
        else:
            w_s, h_s, theta_w, theta_h = ff.fgd_update(v_s, w_s, h_s, theta_w, theta_h)
        h_os = ff.calculate_h(v, w_s, print_interim=False)
        snmf_error[i] = np.linalg.norm(v - np.dot(w_s, h_os)) ** 2
        if (i == 0) | (i % 100 == 99):
            print("SketchingNMF ( r={}  k={} seed={} ) : {} times update".format(r, approximate_size, wh_seed, i + 1))
    if t_flag & snmf_only:
        return nmf_error, snmf_error, None, None, h_os.T, w_s.T, h_s.T
    elif t_flag:
        return nmf_error, snmf_error, h.T, w.T, h_os.T, w_s.T, h_s.T
    elif snmf_only:
        return nmf_error, snmf_error, None, None, w_s, h_os, h_s
    else:
        return nmf_error, snmf_error, w, h, w_s, h_os, h_s
