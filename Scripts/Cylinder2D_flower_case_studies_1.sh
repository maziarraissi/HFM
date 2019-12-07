#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Cylinder2D_flower_systematic.py 201 15000 > Cylinder2D_flower_201_15000_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D_flower_systematic.py 101 15000 > Cylinder2D_flower_101_15000_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Cylinder2D_flower_systematic.py 51 15000 > Cylinder2D_flower_51_15000_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_flower_systematic.py 26 15000 > Cylinder2D_flower_26_15000_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Cylinder2D_flower_systematic.py 201 10000 > Cylinder2D_flower_201_10000_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Cylinder2D_flower_systematic.py 101 10000 > Cylinder2D_flower_101_10000_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder2D_flower_systematic.py 51 10000 > Cylinder2D_flower_51_10000_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower_systematic.py 26 10000 > Cylinder2D_flower_26_10000_stdout.txt&
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_flower_systematic.py 201 5000 > Cylinder2D_flower_201_5000_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_systematic.py 101 5000 > Cylinder2D_flower_101_5000_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Cylinder2D_flower_systematic.py 51 5000 > Cylinder2D_flower_51_5000_stdout.txt &
CUDA_VISIBLE_DEVICES=11 python Cylinder2D_flower_systematic.py 26 5000 > Cylinder2D_flower_26_5000_stdout.txt &
CUDA_VISIBLE_DEVICES=12 python Cylinder2D_flower_systematic.py 201 2500 > Cylinder2D_flower_201_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=13 python Cylinder2D_flower_systematic.py 101 2500 > Cylinder2D_flower_101_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=14 python Cylinder2D_flower_systematic.py 51 2500 > Cylinder2D_flower_51_2500_stdout.txt &
CUDA_VISIBLE_DEVICES=15 python Cylinder2D_flower_systematic.py 26 2500 > Cylinder2D_flower_26_2500_stdout.txt &

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
