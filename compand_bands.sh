#!/bin/bash
curr_arg=0
attack_delay=()
for arg in "$@"; do
	if [ $curr_arg -eq 0 ]; then
		output_path=$arg
	elif [ $curr_arg -eq 1 ]; then
		comp_lvl=$arg
	else
		attack_delay+=($arg)
	fi
	curr_arg=$((curr_arg+1))
done

rm "${output_path}"soxtmpcomp_*.wav

num_bands=${#attack_delay[@]}
for (( i=0 ; i<$num_bands ; i++ )); do
	if [ $comp_lvl -eq 0 ]; then
		sox "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-70,-60,-24 -0 -90 norm
	elif [ $comp_lvl -eq 1 ]; then
		sox "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-70,-60,-12 -0 -90 norm
	elif [ $comp_lvl -eq 2 ]; then
		sox "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-70,-60,-6 -0 -90 norm
	elif [ $comp_lvl -eq 3 ]; then
		sox "${output_path}soxtmp_$i.wav" "${output_path}soxtmpcomp_$i.wav" norm compand ${attack_delay[$i]} 6:-70,-60,-3 -0 -90 norm
	fi
done

