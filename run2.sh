#!/bin/bash

let "n = $(python script.py)"

for ((i=0; i<n; i++))
        do
		for ((j=0; j<i; j++))
			do
				python automate.py $j $i
				python wrapper.py -c
				python wrapper.py -tp
			done
	done
