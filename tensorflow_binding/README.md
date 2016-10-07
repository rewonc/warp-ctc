g++ -Wall -std=c++11 -shared warpctc/warp.cc -o warpctc/warp.so -fPIC -I $TF_INC -I "/mnt/home/rewon/dev/warp-ctc/include/" -L/mnt/home/rewon/dev/warp-ctc/build -lwarpctc -g

LD_LIBRARY_PATH=/mnt/home/rewon/dev/warp-ctc/build:$LD_LIBRARY_PATH py.test

TODO: you can't do any operations on the gpu tensor.
Need to use Eigen's GPU things?
See the functor examples...
https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/core/kernels/pad_op.cc