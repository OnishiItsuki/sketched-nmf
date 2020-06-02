#!/bin/bash
trap 'docker stop $(docker ps -q -f ancestor=repo-luna.ist.osaka-u.ac.jp:5000/ionihsi/anaconda3-cuda:9.0-cudnn7-devel-ubuntu16.04 -f name=parallel_container_r)' EXIT

for r_num in 50
do
for ap_num in 250 500 750 1000 1250 1500 1750 2000 2250 2429
# for ap_num in 10 20
   do
   docker run -t --rm \
      --name parallel_container_r${r_num}_ap${ap_num} \
      --hostname anaconda3-cuda-ionishi \
      -v /mnt/workspace2019/ionishi:/home/ionishi/mnt/workspace \
      -v ~/:/home/ionishi \
      -e "LANG=ja_JP.UTF-8" \
      -e "TIMEZONE=Asia/Tokyo" \
      -e OMP_NUM_THREADS="1" \
      repo-luna.ist.osaka-u.ac.jp:5000/ionihsi/anaconda3-cuda:9.0-cudnn7-devel-ubuntu16.04 \
      bash /home/ionishi/sketching-nmf/bashfile/run.sh $r_num $ap_num &
   done
done
