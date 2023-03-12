#!/bin/env bash


mkdir -p img

if [ ! -n "$(find img -name '*.png')" ]; then
    # Parse files and collects unique names ending with .png
    for img in $(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub | sort | uniq)
    do 
        echo ${img}
        wget -q https://nvlabs.github.io/cub/${img} -O img/${img}
    done  
fi

./repo.sh docs
