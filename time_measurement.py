import time
import numpy as np
import functionfile as ff


def time_measurement(n, m, r, approximate_size, v, iteration, seeds, c_mode, column_sketching):
    # NMF------------------------------------
    print("\n\n\n------------------   NMF   -------------------")

    start = time.time()
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)

    for i in range(iteration):
        w, h = ff.update(v, w, h, c_mode)

    elapsed_time = time.time() - start
    print("\nNMF time: " + str(elapsed_time))
    t_result = np.array(elapsed_time)

    # Sketching NMF --------------------------
    print("\n\n-------------   Sketching NMF   --------------")

    start = time.time()
    v_s = ff.uniform_sampling(v, approximate_size, seeds[0] + 1, right_product=column_sketching)

    if column_sketching:
        w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
        h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)
        for i in range(iteration):
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
        h_os = ff.calculate_h(v, w_s, print_interim=True)

    else:
        w_s = ff.generate_w(approximate_size, r, seeds[0], c_mode=c_mode)
        h_s = ff.generate_h(r, m, seeds[1], c_mode=c_mode)
        for i in range(iteration):
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
        w_os = ff.calculate_h(v, h_s, print_interim=True)

    elapsed_time = time.time() - start
    print("\nSketching NMF time: " + str(elapsed_time) + "\n")
    t_result = np.append(t_result, elapsed_time)

    return t_result, w, h, w_s, h_s


def parallel_time_measurement(r, approximate_size, v, iteration, wh_seed, c_mode, t_flag=False, snmf_only=False, output_matrices=False):
    n, m = v.shape
    seeds = ff.two_seeds(wh_seed)

    # NMF  ---------------------------------------------------------------------------------------------------------
    if not snmf_only:
        start = time.time()
        w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
        h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)

        if c_mode == 2:
            theta_w = theta_h = 5
            for i in range(iteration):
                w, h, theta_w, theta_h = ff.fgd_update(v, w, h, theta_w, theta_h)
        else:
            for i in range(iteration):
                w, h = ff.update(v, w, h, c_mode)
        t_result = np.array(time.time() - start)
    else:
        t_result = [None]

    # Sketching NMF ------------------------------------------------------------------------------------------------
    start = time.time()
    v_s = ff.uniform_sampling(v, approximate_size, seeds[0] + 1)
    w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

    if c_mode == 2:
        theta_w = theta_h = 5
        for i in range(iteration):
            w_s, h_s, theta_w, theta_h = ff.fgd_update(v_s, w_s, h_s, theta_w, theta_h)
    else:
        for i in range(iteration):
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)
    h_os = ff.calculate_h(v, w_s, print_interim=False)

    t_result = np.append(t_result, time.time() - start)

    if output_matrices:
        if t_flag & snmf_only:
            return t_result, None, None, h_os.T, w_s.T, h_s.T
        elif t_flag:
            return t_result, h.T, w.T, h_os.T, w_s.T, h_s.T
        elif snmf_only:
            return t_result, None, None, w_s, h_os, h_s
        else:
            return t_result, w, h, w_s, h_os, h_s
    else:
        return t_result
