#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic_noise.py 0.98 > Cylinder2D_flower_noise_98_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic_noise.py 1.00 > Cylinder2D_flower_noise_100_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic_noise.py 1.02 > Cylinder2D_flower_noise_102_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic_noise.py 1.04 > Cylinder2D_flower_noise_104_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic_noise.py 1.06 > Cylinder2D_flower_noise_106_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic_noise.py 1.08 > Cylinder2D_flower_noise_108_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic_noise.py 1.10 > Cylinder2D_flower_noise_110_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic_noise.py 1.12 > Cylinder2D_flower_noise_112_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic_noise.py 1.14 > Cylinder2D_flower_noise_114_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic_noise.py 1.16 > Cylinder2D_flower_noise_116_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic_noise.py 1.18 > Cylinder2D_flower_noise_118_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic_noise.py 1.20 > Cylinder2D_flower_noise_120_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic_noise.py 1.22 > Cylinder2D_flower_noise_122_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic_noise.py 1.24 > Cylinder2D_flower_noise_124_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic_noise.py 1.26 > Cylinder2D_flower_noise_126_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic_noise.py 1.28 > Cylinder2D_flower_noise_128_stdout.txt &

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
