#!/bin/bash
input_path=$1
output_path=$2
comp_lvl=$3
#buffer=262144
buffer=524288
#buffer=1048576

if [[ "$comp_lvl" == "x"  ]]; then
	# 1:1
	cp "${input_path}" "${output_path}x--.wav"
elif [[ $comp_lvl -eq 0  ]]; then
	# 1.2:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}0--.wav" norm compand 0.3,1 6:-70,-60,-50 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 1  ]]; then
	# 1.5:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}1--.wav" norm compand 0.3,1 6:-70,-60,-40 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 2  ]]; then
	# 2:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}2--.wav" norm compand 0.3,1 6:-70,-60,-30 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 3  ]]; then
	# 3:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}3--.wav" norm compand 0.3,1 6:-70,-60,-20 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 4  ]]; then
	# 4:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}4--.wav" norm compand 0.3,1 6:-70,-60,-15 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 5  ]]; then
	# 5:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}5--.wav" norm compand 0.3,1 6:-70,-60,-12 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 6  ]]; then
	# 8:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}6--.wav" norm compand 0.3,1 6:-70,-60,-7.5 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 7  ]]; then
	# 10:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}7--.wav" norm compand 0.3,1 6:-70,-60,-6 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 8  ]]; then
	# 12:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}8--.wav" norm compand 0.3,1 6:-70,-60,-5 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 9  ]]; then
	# 20:1
	sox --buffer $buffer --multi-threaded "${input_path}" "${output_path}8--.wav" norm compand 0.3,1 6:-70,-60,-3 -0 -90 0.2 norm
fi
