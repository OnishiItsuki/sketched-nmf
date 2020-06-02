#!/bin/bash
while :
do
tmp=($(docker stop $(docker ps -q -f ancestor=repo-luna.ist.osaka-u.ac.jp:5000/ionihsi/anaconda3-cuda:9.0-cudnn7-devel-ubuntu16.04 -f name=parallel_container_)))

echo "${tmp[@]}"

if [ -n ${tmp[@]} ] ; then
 echo "stop all container"
 break
fi

echo "now stoping container"

sleep 10s
done
