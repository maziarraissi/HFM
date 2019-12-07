#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic.py 26 2500 > Cylinder2D_flower_26_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic.py 13 2500 > Cylinder2D_flower_13_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic.py 7 2500 > Cylinder2D_flower_7_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic.py 3 2500 > Cylinder2D_flower_3_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic.py 26 1500 > Cylinder2D_flower_26_1500_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic.py 13 1500 > Cylinder2D_flower_13_1500_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic.py 7 1500 > Cylinder2D_flower_7_1500_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic.py 3 1500 > Cylinder2D_flower_3_1500_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic.py 26 500 > Cylinder2D_flower_26_500_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic.py 13 500 > Cylinder2D_flower_13_500_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic.py 7 500 > Cylinder2D_flower_7_500_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic.py 3 500 > Cylinder2D_flower_3_500_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic.py 26 250 > Cylinder2D_flower_26_250_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic.py 13 250 > Cylinder2D_flower_13_250_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic.py 7 250 > Cylinder2D_flower_7_250_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic.py 3 250 > Cylinder2D_flower_3_250_stdout.txt &

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
