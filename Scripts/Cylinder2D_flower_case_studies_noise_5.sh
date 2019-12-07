#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic_noise.py 1.30 > Cylinder2D_flower_noise_130_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic_noise.py 1.32 > Cylinder2D_flower_noise_132_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic_noise.py 1.34 > Cylinder2D_flower_noise_134_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic_noise.py 1.36 > Cylinder2D_flower_noise_136_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic_noise.py 1.38 > Cylinder2D_flower_noise_138_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic_noise.py 1.40 > Cylinder2D_flower_noise_140_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic_noise.py 1.42 > Cylinder2D_flower_noise_142_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic_noise.py 1.44 > Cylinder2D_flower_noise_144_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic_noise.py 1.46 > Cylinder2D_flower_noise_146_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic_noise.py 1.48 > Cylinder2D_flower_noise_148_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic_noise.py 1.50 > Cylinder2D_flower_noise_150_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic_noise.py 1.52 > Cylinder2D_flower_noise_152_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic_noise.py 1.54 > Cylinder2D_flower_noise_154_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic_noise.py 1.56 > Cylinder2D_flower_noise_156_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic_noise.py 1.58 > Cylinder2D_flower_noise_158_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic_noise.py 1.60 > Cylinder2D_flower_noise_160_stdout.txt &

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
