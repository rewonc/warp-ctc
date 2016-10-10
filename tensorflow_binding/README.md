g++ -Wall -std=c++11 -shared warpctc/warp.cc -o warpctc/warp.so -fPIC -I $TF_INC -I "/mnt/home/rewon/dev/warp-ctc/include/" -L/mnt/home/rewon/dev/warp-ctc/build -lwarpctc -g

LD_LIBRARY_PATH=/mnt/home/rewon/dev/warp-ctc/build:$LD_LIBRARY_PATH py.test

TODO: figure out why eigen operations don't seem to work on GPU. Does the wrapper function need to be written in CUDA?