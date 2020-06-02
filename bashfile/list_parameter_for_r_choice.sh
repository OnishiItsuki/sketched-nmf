#!/bim/bash 
if [ "$2" == "test" ] ;then
 r_list=(5 10)
 seed_list=(1)
elif [ "$1" == "cbcl" -a  "$2" == "run" ] ;then
 r_list=(100 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5)
 seed_list=(1 2 3 4 5 6 7 8 9 10)
elif [ "$1" == "yale" -a  "$2" == "run" ] ;then
 r_list=(100 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5)
 seed_list=(1 2 3 4 5 6 7 8 9 10)
fi

choice_method=("bcv" "bic")
number_of_parallel=$((2*${#r_list[@]}*${#seed_list[@]}))
counter=1
parameter_list=()

for bic_or_bcv in ${choice_method[@]}
do
for r_num in ${r_list[@]}
do
for seed_num in ${seed_list[@]}
do
parameter_list+=($r_num $seed_num $bic_or_bcv $1 $2 $((counter++)) $number_of_parallel)
done
done
done
echo ${parameter_list[@]}
