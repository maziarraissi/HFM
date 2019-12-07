#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic_noise.py 0.64 > Cylinder2D_flower_noise_64_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic_noise.py 0.66 > Cylinder2D_flower_noise_66_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic_noise.py 0.68 > Cylinder2D_flower_noise_68_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic_noise.py 0.70 > Cylinder2D_flower_noise_70_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic_noise.py 0.72 > Cylinder2D_flower_noise_72_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic_noise.py 0.74 > Cylinder2D_flower_noise_74_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic_noise.py 0.76 > Cylinder2D_flower_noise_76_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic_noise.py 0.78 > Cylinder2D_flower_noise_78_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic_noise.py 0.80 > Cylinder2D_flower_noise_80_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic_noise.py 0.82 > Cylinder2D_flower_noise_82_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic_noise.py 0.84 > Cylinder2D_flower_noise_84_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic_noise.py 0.86 > Cylinder2D_flower_noise_86_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic_noise.py 0.88 > Cylinder2D_flower_noise_88_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic_noise.py 0.90 > Cylinder2D_flower_noise_90_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic_noise.py 0.92 > Cylinder2D_flower_noise_92_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic_noise.py 0.94 > Cylinder2D_flower_noise_94_stdout.txt &

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
