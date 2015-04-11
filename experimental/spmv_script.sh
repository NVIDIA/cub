#!/bin/bash



for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216
do
	./bin/spmv_compare_sm520-370-350_nvvm_7.0_abi_nocdp_x86_64 --i=100 --dense=$i --device=$1 $2 $3
done

for i in `ls ../../graphs/*.mtx ../../graphs/spmv/*.mtx`
do 
	./bin/spmv_compare_sm520-370-350_nvvm_7.0_abi_nocdp_x86_64 --i=500 --mtx=$i --device=$1 $2 $3
done 

./bin/spmv_compare_sm520-370-350_nvvm_7.0_abi_nocdp_x86_64 --i=1000 --grid3d=300 --device=$1 $2 $3
./bin/spmv_compare_sm520-370-350_nvvm_7.0_abi_nocdp_x86_64 --i=1000 --grid2d=4000 --device=$1 $2 $3
