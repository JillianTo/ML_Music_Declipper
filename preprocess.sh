#!/bin/bash
mkdir tmp

# Resample to 44.1kHz if not already
for i in *.wav; do
	max=$(ffmpeg -i "$i" -af "volumedetect" -vn -sn -dn 2>&1 | grep "Hz" | sed 's/.*Stream.*), //' | sed 's/\ Hz.*//')
       if [ "$max" != 44100 ]; then
	       ffmpeg -i "$i" -ar "44100" "./tmp/${i%.*}.wav"
       #else
       #	echo "skipping resample"
       fi
done
mv ./tmp/*.wav .

# Normalize peak volume to 0dB
for i in *.wav; do
	max=$(ffmpeg -i "$i" -af "volumedetect" -vn -sn -dn -f null /dev/null 2>&1 | grep "max_volume:" | sed 's/.*max_volume:\ //' | sed 's/\ dB//')
       if [ "$max" != 0.0 ]; then
	       max=$(echo $max | sed 's/-//')
	       ffmpeg -i "$i" -af "volume=${max}dB" "./tmp/${i%.*}.wav"
       fi
done
mv ./tmp/*.wav .

# Run dynaudnorm 
for i in *.wav; do ffmpeg -i "$i" -filter:a "dynaudnorm" "./tmp/${i%.*}.wav"; done
mv ./tmp/*.wav .

# Shorten any silences longer than 2s to 0.5s 
for i in *.wav; do ffmpeg -i "$i" -af "silenceremove=stop_periods=-1:stop_duration=2:stop_threshold=-50dB:stop_silence=0.5" ./tmp/"${i}"; done
mv ./tmp/*.wav .


