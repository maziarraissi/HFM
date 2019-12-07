#!/bin/bash

FAIL=0

CUDA_VISIBLE_DEVICES=0 python Stenosis2D.py > Stenosis2D_stdout.txt &
CUDA_VISIBLE_DEVICES=1 python Cylinder2D.py > Cylinder2D_stdout.txt &
CUDA_VISIBLE_DEVICES=2 python Aneurysm3D.py > Aneurysm3D_stdout.txt &
CUDA_VISIBLE_DEVICES=3 python Cylinder2D_Pec_Re.py > Cylinder2D_Pec_Re_stdout.txt &
CUDA_VISIBLE_DEVICES=4 python Stenosis2D_Pec_Re.py > Stenosis2D_Pec_Re_stdout.txt &
CUDA_VISIBLE_DEVICES=5 python Davinci.py > Davinci_stdout.txt &
CUDA_VISIBLE_DEVICES=6 python Cylinder3D.py > Cylinder3D_stdout.txt &
CUDA_VISIBLE_DEVICES=7 python Cylinder2D_flower.py > Cylinder2D_flower_stdout.txt &
CUDA_VISIBLE_DEVICES=8 python Cylinder2D_No_Slip.py > Cylinder2D_No_Slip_stdout.txt &
CUDA_VISIBLE_DEVICES=9 python Cylinder2D_flower_convergence_plot.py > Cylinder2D_flower_convergence_plot_stdout.txt &
CUDA_VISIBLE_DEVICES=10 python Aneurysm3D_Wall_Stresses.py > Aneurysm3D_Wall_Stresses_stdout.txt &

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
