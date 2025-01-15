#!/bin/bash
input_path=$1
output_path=$2
comp_lvl=$3
buffer=$4

if [[ "$comp_lvl" == "x"  ]]; then
	# 1:1
	#cp "${input_path}" "${output_path}x--.wav"
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm
elif [[ $comp_lvl -eq 0  ]]; then
	# 1.1:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-55 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 1  ]]; then
	# 1.2:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-50 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 2  ]]; then
	# 1.3:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-46 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 3  ]]; then
	# 1.5:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-40 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 4  ]]; then
	# 2:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-30 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 5  ]]; then
	# 3:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-20 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 6  ]]; then
	# 4:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-15 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 7  ]]; then
	# 5:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-12 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 8  ]]; then
	# 8:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-7.5 -0 -90 0.2 norm
elif [[ $comp_lvl -eq 9  ]]; then
	# 10:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-6 -0 -90 0.2 norm
elif [[ $comp_lvl == "A"  ]]; then
	# 12:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-5 -0 -90 0.2 norm
elif [[ "$comp_lvl" == "B"  ]]; then
	# 20:1
	sox -V0 --buffer $buffer --multi-threaded "${input_path}" "${output_path}" norm compand 0.3,1 6:-70,-60,-3 -0 -90 0.2 norm
fi
