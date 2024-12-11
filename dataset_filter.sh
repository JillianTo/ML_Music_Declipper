#!/bin/bash

dr14tmeter_path=/mnt/PC801/Applications/dr14_t.meter-1.0.16
source $dr14tmeter_path/venv/bin/activate

mkdir filtered

for file in ./*.wav; do 
	DR=$(python3 $dr14tmeter_path/dr14tmeter/dr14_tmeter.py -n -f "$file" | grep "DR      = " | sed 's/DR      = //')
	if [ $DR -lt 10 ]; then
		mv "$file" ./filtered
	fi
done

for file in ./*.wav; do 
	DR=$(ffmpeg -i "$file" -af ebur128=framelog=verbose -f null - 2>&1 | awk '/LRA:/{print $2}')
	if (( $(echo "$DR < 3" | bc -l)  )); then
		mv "$file" ./filtered
	fi
done

