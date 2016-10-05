import tensorflow as tf
warp = tf.load_op_library('./warp.so')
with tf.Session():
    warp.ctc([[1, 2], [3, 4]]).eval()



g++ -std=c++11 -shared warpctc/warp.cc -o warpctc/warp.so -fPIC -I $TF_INC
LD_LIBRARY_PATH=/mnt/home/rewon/dev/warp-ctc/build:$LD_LIBRARY_PATH py.test
