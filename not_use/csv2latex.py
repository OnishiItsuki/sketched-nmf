import pandas as pd

file_name = "14_time_MU_[n100, m10000, r50, as1000, ite5000,seed 0, 1]_time"
read_path = "/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/time/14/" + file_name
write_path = "/home/ionishi/mnt/workspace/sketchingNMF/random_matrix/time/14/" + file_name

df = pd.read_csv(read_path + ".csv")
pd.options.display.float_format = '{:.2f}'.format
df.to_latex(write_path + ".tex")
