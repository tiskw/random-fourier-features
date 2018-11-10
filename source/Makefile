libFastInferenceRFF.so: FastInferenceRFF.o
	gcc -std=c99 -shared -Wl,-soname,libFastInferenceRFF.so -o libFastInferenceRFF.so FastInferenceRFF.o

FastInferenceRFF.o: FastInferenceRFF.c
	gcc -std=c99 -c -fPIC -O2 -mtune=native -march=native -ffast-math ${<} -o ${@}

clean:
	rm -f FastInferenceRFF.o libFastInferenceRFF.so

# vim: noexpandtab
