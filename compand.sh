#!/bin/bash
input_path=$1
output_path=$2
comp_lvl=$3
#buffer=262144
buffer=524288
#buffer=1048576

if [[ "$comp_lvl" == "x"  ]]; then
	cp "${input_path}" "${output_path}x--.wav"
elif [[ $comp_lvl -eq 0  ]]; then
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}0--.wav" norm compand 0.3,1 6:-70,-60,-46 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 1  ]]; then
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}1--.wav" norm compand 0.3,1 6:-70,-60,-44 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 2  ]]; then
	# old 0
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}2--.wav" norm compand 0.3,1 6:-70,-60,-40 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 3  ]]; then
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}3--.wav" norm compand 0.3,1 6:-70,-60,-36 -0 -90 0.2 norm
fi
