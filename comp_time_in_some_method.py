import functionfile as ff
import numpy as np
import sys
import time


def get_v_ite(v, n, m, r, approximate_size, seeds, c_mode, CDC, NMFQP=False):
    iteration_list = np.zeros([2], dtype="int")

    # NMF------------------------------------
    print("\n------------------   NMF   -------------------")
    nmf_error = 1000000
    w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)

    if NMFQP:
        print("NMF matrix H is calculated by QP.")
        while nmf_error >= CDC:
            w, h = ff.update(v, w, h, c_mode)

            h_qp = ff.calculate_h(v, w, True)
            nmf_error = np.linalg.norm(v - np.dot(w, h_qp)) ** 2

            if (iteration_list[0] == 0) | (iteration_list[0] % 100 == 99):
                print(str(iteration_list[0] + 1) + " times update  error: " + str(nmf_error))
            iteration_list[0] += 1

    else:
        while nmf_error >= CDC:
            w, h = ff.update(v, w, h, c_mode)
            nmf_error = np.linalg.norm(v - np.dot(w, h)) ** 2

            if (iteration_list[0] == 0) | (iteration_list[0] % 100 == 99):
                print(str(iteration_list[0] + 1) + " times update  error: " + str(nmf_error))
            iteration_list[0] += 1

    # Sketching NMF --------------------------
    print("\n-------------   Sketching NMF   --------------")

    snmf_error = 1000000
    v_s = ff.uniform_sampling(v, approximate_size, 0, t=True)
    w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
    h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

    while snmf_error >= CDC:
        w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)

        h_os = ff.calculate_h(v, w_s, True)
        snmf_error = np.linalg.norm(v - np.dot(w_s, h_os)) ** 2

        if (iteration_list[1] == 0) | (iteration_list[1] % 100 == 99):
            print(str(iteration_list[1] + 1) + " times update  error: " + str(snmf_error))
        iteration_list[1] = iteration_list[1] + 1

    print(iteration_list)
    return iteration_list


def time_measurement(n, m, r, approximate_size, v, iteration, seeds, c_mode, NMF_or_SNMF):
    if NMF_or_SNMF == 0:
        # NMF------------------------------------
        print("\n\n\n------------------   NMF   -------------------")

        start = time.time()
        w = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
        h = ff.generate_h(r, m, seeds[1], c_mode=c_mode)

        for i in range(0, iteration):
            w, h = ff.update(v, w, h, c_mode)

        t_result = time.time() - start
        print("\nNMF time: " + str(t_result))

    else:
        # Sketching NMF --------------------------
        print("\n\n-------------   Sketching NMF   --------------")

        start = time.time()
        v_s = ff.uniform_sampling(v, approximate_size, 0, t=True)
        w_s = ff.generate_w(n, r, seeds[0], c_mode=c_mode)
        h_s = ff.generate_h(r, approximate_size, seeds[1], c_mode=c_mode)

        for i in range(0, iteration):
            w_s, h_s = ff.update(v_s, w_s, h_s, c_mode)

        _ = ff.calculate_h(v, w_s, True)

        t_result = time.time() - start
        print("\nSketching NMF time: " + str(t_result) + "\n")

    return t_result
