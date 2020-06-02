#!/bin/bash
r=25
ite=3000

# for i in 250 500 750 1000 1250 1500 1750 2000 2250 2429
for i in 77760
do
for j in 1 2 3 4 5 6 7 8 9 10
do
rm /home/ionishi/mnt/workspace/sketchingNMF/CBCL/train/9\(for_thesis\)/MU/r\,k/r\=50/k\=${i}/time/realdata_9\(for_thesis\)_V_MU_\[n361\,m2429\,r50\,as${i}\,ite3000\,seed0\,${j}\].csv
rm /home/ionishi/mnt/workspace/sketchingNMF/CBCL/train/9\(for_thesis\)/HALS/r\,k/r\=50/k\=${i}/time/realdata_9\(for_thesis\)_V_HALS_\[n361\,m2429\,r50\,as${i}\,ite3000\,seed0\,${j}\].csv
rm /home/ionishi/mnt/workspace/sketchingNMF/YaleFD/faces/9\(for_thesis\)/MU/r\,k/r\=25/k\=${i}/time/realdata_9\(for_thesis\)_V_MU_\[n165\,m77760\,r25\,as${i}\,ite3000\,seed0\,${j}\].csv
rm /home/ionishi/mnt/workspace/sketchingNMF/Yale/faces/9\(for_thesis\)/HALS/r\,k/r\=25/k\=${i}/time/realdata_9\(for_thesis\)_V_HALS_\[n165\,m77760\,r25\,as${i}\,ite3000\,seed0\,${j}\].csv
done
done
