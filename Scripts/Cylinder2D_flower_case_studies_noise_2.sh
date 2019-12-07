#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic_noise.py 0.32 > Cylinder2D_flower_noise_32_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic_noise.py 0.34 > Cylinder2D_flower_noise_34_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic_noise.py 0.36 > Cylinder2D_flower_noise_36_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic_noise.py 0.38 > Cylinder2D_flower_noise_38_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic_noise.py 0.40 > Cylinder2D_flower_noise_40_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic_noise.py 0.42 > Cylinder2D_flower_noise_42_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic_noise.py 0.44 > Cylinder2D_flower_noise_44_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic_noise.py 0.46 > Cylinder2D_flower_noise_46_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic_noise.py 0.48 > Cylinder2D_flower_noise_48_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic_noise.py 0.50 > Cylinder2D_flower_noise_50_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic_noise.py 0.52 > Cylinder2D_flower_noise_52_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic_noise.py 0.54 > Cylinder2D_flower_noise_54_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic_noise.py 0.56 > Cylinder2D_flower_noise_56_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic_noise.py 0.58 > Cylinder2D_flower_noise_58_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic_noise.py 0.60 > Cylinder2D_flower_noise_60_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic_noise.py 0.62 > Cylinder2D_flower_noise_62_stdout.txt &

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
