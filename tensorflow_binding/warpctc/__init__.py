import tensorflow as tf
from tensorflow.python.framework import ops
import os


_warpctc = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'warp.so'))


def ctc(inputs, input_lens, labels, label_lens):
    loss, _ = _warpctc.warp_ctc(inputs, input_lens, labels, label_lens)
    return loss


@ops.RegisterGradient("WarpCTC")
def _CTCLossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    return [grad, None, None, None]


@ops.RegisterShape("WarpCTC")
def _CTCLossShape(op):
    inputs_shape = op.inputs[0].get_shape().with_rank(3)
    batch_size = op.inputs[0].get_shape()[0]
    return [batch_size, inputs_shape]
