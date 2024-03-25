#!/bin/bash
input_path=$1
output_path=$2
comp_lvl=$3

if [ $comp_lvl -ne 0 ]; then
	#mv "${input_path}" "${output_path}0--.wav"
	sox "${input_path}" "${output_path}0--.wav" norm compand 0.3,1 6:-70,-60,-40 -0 -90 0.2 norm
fi
if [ $comp_lvl -ne 1 ]; then
	sox "${input_path}" "${output_path}1--.wav" norm compand 0.3,1 6:-70,-60,-38 -0 -90 0.2 norm
fi

rm "${input_path}"

