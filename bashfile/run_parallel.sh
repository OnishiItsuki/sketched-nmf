#!/bin/bash
if [ "$1" == "random" ] ;then
bash list_parameter_for_docker_run_p.sh $1 $2 | xargs -n 9 -P $[$(grep processor /proc/cpuinfo | \
sed 's/[^0-9]//g'| sort -nr | head -n 1) - 3] bash docker_run_parallel_for_random.sh

elif [ "$1" == "cbcl" -o "$1" == "yale" ] ;then
bash list_parameter_for_docker_run_p.sh $1 $2 | xargs -n 7 -P $[$(grep processor /proc/cpuinfo | \
sed 's/[^0-9]//g'| sort -nr | head -n 1) - 3] bash docker_run_parallel.sh
fi
