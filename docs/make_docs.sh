#!/usr/bin/env bash
./update.sh
make html

for ext in png jpg gif pdf; do
	echo ../notebooks/tutorials/*.$ext ./tutorials/
	cp ../notebooks/tutorials/*.$ext ./tutorials/
	for fils in tutorials _images ; do
		echo ../notebooks/tutorials/*.$ext _build/html/$fils/
		cp ../notebooks/tutorials/*.$ext _build/html/$fils/
	done
done
