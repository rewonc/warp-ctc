import tensorflow as tf
import os

warp = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'warp.so'))
