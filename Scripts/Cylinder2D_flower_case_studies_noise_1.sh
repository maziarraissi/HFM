#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic_noise.py 0.01 > Cylinder2D_flower_noise_01_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic_noise.py 0.02 > Cylinder2D_flower_noise_02_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic_noise.py 0.04 > Cylinder2D_flower_noise_04_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic_noise.py 0.06 > Cylinder2D_flower_noise_06_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic_noise.py 0.08 > Cylinder2D_flower_noise_08_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic_noise.py 0.10 > Cylinder2D_flower_noise_10_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic_noise.py 0.12 > Cylinder2D_flower_noise_12_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic_noise.py 0.14 > Cylinder2D_flower_noise_14_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic_noise.py 0.16 > Cylinder2D_flower_noise_16_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic_noise.py 0.18 > Cylinder2D_flower_noise_18_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic_noise.py 0.20 > Cylinder2D_flower_noise_20_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic_noise.py 0.22 > Cylinder2D_flower_noise_22_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic_noise.py 0.24 > Cylinder2D_flower_noise_24_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic_noise.py 0.26 > Cylinder2D_flower_noise_26_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic_noise.py 0.28 > Cylinder2D_flower_noise_28_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic_noise.py 0.30 > Cylinder2D_flower_noise_30_stdout.txt &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi
