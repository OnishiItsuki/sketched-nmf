#!/bin/bash
docker run -t --rm \
   --name parallel_container_r${1}_ap${2}_seed${3} \
   --hostname anaconda3-cuda-ionishi \
   -v /mnt/workspace2019/ionishi:/home/ionishi/mnt/workspace \
   -v ~/:/home/ionishi \
   -e "LANG=ja_JP.UTF-8" \
   -e "TIMEZONE=Asia/Tokyo" \
   -e OMP_NUM_THREADS="1" \
   repo-luna.ist.osaka-u.ac.jp:5000/ionihsi/anaconda3-cuda:9.0-cudnn7-devel-ubuntu16.04 \
   bash /home/ionishi/sketching-nmf/bashfile/run_python.sh $1 $2 $3 $4 $5 $6 $7
