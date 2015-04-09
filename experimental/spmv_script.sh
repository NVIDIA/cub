#!/bin/bash

for i in `ls ../../graphs/*.mtx ../../graphs/spmv/*.mtx`
do 
#echo ------------------------------------------------------------------------------- 
#echo $i 
#echo
./bin/spmv_compare_sm350-520_nvvm_7.0_abi_nocdp_x86_64 --i=2500 --mtx=$i --device=$2 $3 $4
done > $1$2$3$4 2>&1 

./bin/spmv_compare_sm350-520_nvvm_7.0_abi_nocdp_x86_64 --i=2500 --grid3d=300 --device=$2 $3 $4 >> $1$2$3$4 2>&1
./bin/spmv_compare_sm350-520_nvvm_7.0_abi_nocdp_x86_64 --i=2500 --grid2d=4000 --device=$2 $3 $4 >> $1$2$3$4 2>&1
./bin/spmv_compare_sm350-520_nvvm_7.0_abi_nocdp_x86_64 --i=500 --wheel=1000000 --device=$2 $3 $4 >> $1$2$3$4 2>&1