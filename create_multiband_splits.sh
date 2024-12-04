#!/bin/bash
curr_arg=0
xo_freqs=()
for arg in "$@"; do
	if [ $curr_arg -eq 0 ]; then
		input=$arg
	elif [ $curr_arg -eq 1 ]; then
		output_path=$arg
	else
		xo_freqs+=($arg)
	fi
	curr_arg=$((curr_arg+1))
done

rm "${output_path}"soxtmp_*.wav
num_freqs=${#xo_freqs[@]}
for (( i=0 ; i<=$num_freqs ; i++ )); do
	if [ $i -eq 0 ]; then
		tmp_input=$input
	else
		tmp_input="${output_path}soxtmphp_${i}.wav"
	fi
	if [ $i -gt 0 ]; then
		sox -V0 "${input}" "${tmp_input}" highpass ${xo_freqs[$((i-1))]} highpass ${xo_freqs[$((i-1))]}
	fi
	if [ $i -lt $num_freqs ]; then
		sox -V0 "${tmp_input}" "${output_path}soxtmp_${i}.wav" lowpass ${xo_freqs[$i]} lowpass ${xo_freqs[$i]}
	fi
done

mv "${output_path}"soxtmphp_$num_freqs.wav "${output_path}"soxtmp_$num_freqs.wav 
rm "${output_path}"soxtmphp_*.wav
