#!/bin/bash
if [ "$2" == "random" ]
then
#n_list=(77760)
n_list=(165 1000)
#m_list=(165)
m_list=(77760 30000 10000)
r_list=(50)
#r_list=(50)
ap_list=(10000 5000 1000)
#ap_list=(10000 1000 1500 1000 500 1000 1000)
seed_list=(1 2 3 4 5 6 7 8 9 10)
#seed_list=(1)

elif [ "$2" == "cbcl" ]
then
r_list=(50)
#r_list=(50)
ap_list=(2429 2250 2000 1750 1500 1250 1000 750 500 250)
#ap_list=(2429 2250 2000 1750 1500 1250 1000 750 500 250)
seed_list=(1 2 3 4 5 6 7 8 9 10)
#seed_list=(1)

elif [ "$2" == "yale" ]
then
# r_list=(25 50)
r_list=(25)
ap_list=(77760 50000 30000 10000 7500 5000 2500 1000)
#ap_list=(77760 50000 30000 10000 7500 5000 2500 1000)
seed_list=(1 2 3 4 5 6 7 8 9 10)
#seed_list=(1)

elif [ "$2" == "test" ]
then
n_list=(100)
m_list=(10000)
r_list=(50)
ap_list=(1000)
seed_list=(1)
fi

if [ "$1" == "r" ]
then
 echo ${r_list[@]}

elif [ "$1" == "ap" ]
then
 echo ${ap_list[@]}

elif [ "$1" == "seed" ]
then
 echo ${seed_list[@]}

elif [ "$1" == "n" ]
then
 echo ${n_list[@]}

elif [ "$1" == "m" ]
then
 echo ${m_list[@]}
fi
