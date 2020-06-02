#!/bin/bash
bash list_parameter_for_r_choice.sh $1 $2 | xargs -n 7 -P $[$(grep processor /proc/cpuinfo | \
sed 's/[^0-9]//g'| sort -nr | head -n 1) - 3] bash docker_run_parallel_for_r_choice.sh

#bash list_parameter_for_r_choice.sh $1 $2 | xargs -n 7 -P $[$(grep processor /proc/cpuinfo | \
#sed 's/[^0-9]//g'| sort -nr | head -n 1) - 3] bash docker_run_image_print_parallel.sh
