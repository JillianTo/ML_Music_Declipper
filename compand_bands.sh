#!/bin/bash
curr_arg=0
attack_delay=()
for arg in "$@"; do
	if [ $curr_arg -eq 0 ]; then
		output_path=$arg
	elif [ $curr_arg -eq 1 ]; then
		comp_lvl=$arg
	elif [ $curr_arg -eq 2 ]; then
		buffer=$arg
	else
		attack_delay+=($arg)
	fi
	curr_arg=$((curr_arg+1))
done

rm "${output_path}"soxtmpcomp_*.wav

num_bands=${#attack_delay[@]}
for (( i=0 ; i<$num_bands ; i++ )); do
	if [[ "$comp_lvl" == "x"  ]]; then
		# 1:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm
	elif [[ $comp_lvl -eq 0 ]]; then
		# 1.1:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-27 -0 -90 norm
	elif [[ $comp_lvl -eq 1 ]]; then
		# 1.2:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-25 -0 -90 norm
	elif [[ $comp_lvl -eq 2 ]]; then
		# 1.3:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-23 -0 -90 norm
	elif [[ $comp_lvl -eq 3 ]]; then
		# 1.5:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-20 -0 -90 norm
	elif [[ $comp_lvl -eq 4 ]]; then
		# 2:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-15 -0 -90 norm
	elif [[ $comp_lvl -eq 5 ]]; then
		# 3:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-10 -0 -90 norm
	elif [[ $comp_lvl -eq 6 ]]; then
		# 4:1 
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-7.5 -0 -90 norm
	elif [[ $comp_lvl -eq 7 ]]; then
		# 5:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-6 -0 -90 norm
	elif [[ $comp_lvl -eq 8 ]]; then
		# 8:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-3.75 -0 -90 norm
	elif [[ $comp_lvl -eq 9 ]]; then
		# 10:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-3 -0 -90 norm
	elif [[ $comp_lvl == "A" ]]; then
		# 12:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-2.5 -0 -90 norm
	elif [[ "$comp_lvl" == "B" ]]; then
		# 20:1
		sox -V0 --buffer $buffer --multi-threaded "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-40,-30,-1.5 -0 -90 norm
	fi
done

