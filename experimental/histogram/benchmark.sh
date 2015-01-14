#!/bin/sh
echo ",gmem atomics,smem atomics,no atomics,warp coalescing,cub atomics,cub sorting"
echo -n "random,"
./histogram benchmark/random.bin bench
echo -n "nature,"
./histogram benchmark/nature.png bench
echo -n "operahouse,"
./histogram benchmark/operahouse.png bench
echo -n "cityscape,"
./histogram benchmark/cityscape.png bench
echo -n "cheetah,"
./histogram benchmark/cheetah.png bench
echo -n "apples,"
./histogram benchmark/apples.png bench
echo -n "black,"
./histogram benchmark/black.png bench
