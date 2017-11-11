#!/bin/bash

let "n = $(python script.py)"

for ((i=0; i<n; i++))
        do
		python automate.py $i -1
		python wrapper.py -c
		python wrapper.py -tp
	done
