import tensorflow as tf
warp = tf.load_op_library('./warp.so')
with tf.Session():
    warp.ctc([[1, 2], [3, 4]]).eval()

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared warpctc/warp.cc -o warpctc/warp.so -fPIC -I $TF_INC
py.test
