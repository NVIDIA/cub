#!/bin/bash

#for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216
#do
#	$1 --dense=$i $2 $3 $4 $5
#done

echo

for i in `ls /scratch/dumerrill/graphs/mtx/*.mtx`
#for i in `ls /cygdrive/w/Dev/UFget/mtx/*.mtx`
do 
	$1 --mtx=$i $2 $3 $4 $5
done 

#$1 --grid3d=300 $2 $3 $4 $5
#$1 --grid2d=4000 $2 $3 $4 $5
