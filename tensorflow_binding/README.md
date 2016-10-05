g++ -std=c++11 -shared warpctc/warp.cc -o warpctc/warp.so -fPIC -I $TF_INC
LD_LIBRARY_PATH=/mnt/home/rewon/dev/warp-ctc/build:$LD_LIBRARY_PATH py.test
