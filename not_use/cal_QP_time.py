import functionfile as ff
import pandas as pd
import numpy as np

n = 100
m = 10000
r = 50
k_list = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
seed_list = range(1, 11)
Program_num = "15(for_graduate_thesis)"

nmf_tmp = np.zeros([len(k_list), len(seed_list)])
snmf_tmp = np.zeros([len(k_list), len(seed_list)])

for k_i, k in enumerate(k_list):
    for seed_i, seed in enumerate(seed_list):
        nmf_tmp[k_i, seed_i], snmf_tmp[k_i, seed_i] = ff.read_time(r, k, 0, seed, "random_matrix", Program_num, "MU", True)
nmf = np.mean(nmf_tmp, axis=1)
snmf = np.mean(snmf_tmp, axis=1)
df_nmf = pd.DataFrame(nmf_tmp)
df_nmf.to_csv("/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/{}/MU_iteration0/nmf_time.csv".format(Program_num))
df_snmf = pd.DataFrame(snmf_tmp)
df_snmf.to_csv("/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/{}/MU_iteration0/snmf_time.csv".format(Program_num))

result = snmf - nmf
df = pd.DataFrame(result.reshape(1, len(result)), columns=k_list)
df.to_csv("/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/{}/MU_iteration0/QP_time.csv".format(Program_num))
