#!/bin/bash
if [ "$2" == "test" ] ;then
 n_list=($(bash list_parameter.sh n $2))
 m_list=($(bash list_parameter.sh m $2))
 r_list=($(bash list_parameter.sh r $2))
 ap_list=($(bash list_parameter.sh ap $2))
 seed_list=($(bash list_parameter.sh seed $2))
elif [ "$2" == "run" ] ;then
 if [ "$1" == "random" ] ;then
  n_list=($(bash list_parameter.sh n $1))
  m_list=($(bash list_parameter.sh m $1))
 fi
 r_list=($(bash list_parameter.sh r $1))
 ap_list=($(bash list_parameter.sh ap $1))
 seed_list=($(bash list_parameter.sh seed $1))
fi

if [ "$1" == "random" ] ;then
number_of_parallel=$((${#n_list[@]}*${#m_list[@]}*${#r_list[@]}*${#ap_list[@]}*${#seed_list[@]}))
else
number_of_parallel=$((${#r_list[@]}*${#ap_list[@]}*${#seed_list[@]}))
fi
counter=1
parameter_list=()
if [ "$1" == "random" ] ;then
#for ((j=0;j<${#n_list[@]};j++))
#do
for n_num in ${n_list[@]}
do
for m_num in ${m_list[@]}
do
for r_num in ${r_list[@]}
do
for ap_num in ${ap_list[@]}
do
for seed_num in ${seed_list[@]}
do
parameter_list+=($r_num $ap_num $seed_num $1 $2 $((counter++)) $number_of_parallel $n_num $m_num)
done
done
done
done
done
#done

else
for r_num in ${r_list[@]}
do
for ap_num in ${ap_list[@]}
do
for seed_num in ${seed_list[@]}
do
parameter_list+=($r_num $ap_num $seed_num $1 $2 $((counter++)) $number_of_parallel)
done
done
done
fi
echo ${parameter_list[@]}
