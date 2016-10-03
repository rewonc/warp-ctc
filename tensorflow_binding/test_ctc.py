import tensorflow as tf
import numpy as np
from warpctc import ctc


def get_dense_array(labels):
    '''Returns a dense np array of labels and lengths'''
    # indices should be one-indexed.
    labels_1 = [[x + 1 for x in li] for li in labels]
    lengths = [len(li) for li in labels]
    max_len = max(lengths)
    n_instances = len(labels)
    arr = np.zeros([n_instances, max_len], dtype=np.int32)
    for i in range(n_instances):
        l = lengths[i]
        arr[i, 0:l] = labels_1[i]
    return arr, np.array(lengths, dtype=np.int32)


class CTCLossTest(tf.test.TestCase):

    def _testCTCLoss(self, inputs, input_lens, labels, label_lens,
                     loss_truth, grad_truth, expected_err_re=None):
        self.assertEquals(inputs.shape, grad_truth.shape)

        inputs_t = tf.constant(inputs)

        with self.test_session(use_gpu=False) as sess:
            loss = ctc(inputs=inputs_t, input_lens=input_lens,
                       labels=labels, label_lens=label_lens)
            grad = tf.gradients(loss, [inputs_t])[0]

            import pdb; pdb.set_trace()

            self.assertShapeEqual(loss_truth, loss)
            self.assertShapeEqual(grad_truth, grad)

            if expected_err_re is None:
                (tf_loss, tf_grad) = sess.run([loss, grad])
                self.assertAllClose(tf_loss, loss_truth, atol=1e-6)
                self.assertAllClose(tf_grad, grad_truth, atol=1e-6)
            else:
                with self.assertRaisesOpError(expected_err_re):
                    sess.run([loss, grad])

    def testBasic(self):
        """Test two batch entries."""
        # Input and ground truth from Alex Graves' implementation.
        #
        #### Batch entry 0 #####
        # targets: 0 1 2 1 0
        # outputs:
        # 0 0.633766 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
        # 1 0.111121 0.588392 0.278779 0.0055756 0.00569609 0.010436
        # 2 0.0357786 0.633813 0.321418 0.00249248 0.00272882 0.0037688
        # 3 0.0663296 0.643849 0.280111 0.00283995 0.0035545 0.00331533
        # 4 0.458235 0.396634 0.123377 0.00648837 0.00903441 0.00623107
        # alpha:
        # 0 -3.64753 -0.456075 -inf -inf -inf -inf -inf -inf -inf -inf -inf
        # 1 -inf -inf -inf -0.986437 -inf -inf -inf -inf -inf -inf -inf
        # 2 -inf -inf -inf -inf -inf -2.12145 -inf -inf -inf -inf -inf
        # 3 -inf -inf -inf -inf -inf -inf -inf -2.56174 -inf -inf -inf
        # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf -3.34211 -inf
        # beta:
        # 0 -inf -2.88604 -inf -inf -inf -inf -inf -inf -inf -inf -inf
        # 1 -inf -inf -inf -2.35568 -inf -inf -inf -inf -inf -inf -inf
        # 2 -inf -inf -inf -inf -inf -1.22066 -inf -inf -inf -inf -inf
        # 3 -inf -inf -inf -inf -inf -inf -inf -0.780373 -inf -inf -inf
        # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf 0 0
        # prob: -3.34211
        # outputDerivs:
        # 0 -0.366234 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
        # 1 0.111121 -0.411608 0.278779 0.0055756 0.00569609 0.010436
        # 2 0.0357786 0.633813 -0.678582 0.00249248 0.00272882 0.0037688
        # 3 0.0663296 -0.356151 0.280111 0.00283995 0.0035545 0.00331533
        # 4 -0.541765 0.396634 0.123377 0.00648837 0.00903441 0.00623107
        #
        #### Batch entry 1 #####
        #
        # targets: 0 1 1 0
        # outputs:
        # 0 0.30176 0.28562 0.0831517 0.0862751 0.0816851 0.161508
        # 1 0.24082 0.397533 0.0557226 0.0546814 0.0557528 0.19549
        # 2 0.230246 0.450868 0.0389607 0.038309 0.0391602 0.202456
        # 3 0.280884 0.429522 0.0326593 0.0339046 0.0326856 0.190345
        # 4 0.423286 0.315517 0.0338439 0.0393744 0.0339315 0.154046
        # alpha:
        # 0 -1.8232 -1.19812 -inf -inf -inf -inf -inf -inf -inf
        # 1 -inf -2.19315 -2.83037 -2.1206 -inf -inf -inf -inf -inf
        # 2 -inf -inf -inf -2.03268 -3.71783 -inf -inf -inf -inf
        # 3 -inf -inf -inf -inf -inf -4.56292 -inf -inf -inf
        # 4 -inf -inf -inf -inf -inf -inf -inf -5.42262 -inf
        # beta:
        # 0 -inf -4.2245 -inf -inf -inf -inf -inf -inf -inf
        # 1 -inf -inf -inf -3.30202 -inf -inf -inf -inf -inf
        # 2 -inf -inf -inf -inf -1.70479 -0.856738 -inf -inf -inf
        # 3 -inf -inf -inf -inf -inf -0.859706 -0.859706 -0.549337 -inf
        # 4 -inf -inf -inf -inf -inf -inf -inf 0 0
        # prob: -5.42262
        # outputDerivs:
        # 0 -0.69824 0.28562 0.0831517 0.0862751 0.0816851 0.161508
        # 1 0.24082 -0.602467 0.0557226 0.0546814 0.0557528 0.19549
        # 2 0.230246 0.450868 0.0389607 0.038309 0.0391602 -0.797544
        # 3 0.280884 -0.570478 0.0326593 0.0339046 0.0326856 0.190345
        # 4 -0.576714 0.315517 0.0338439 0.0393744 0.0339315 0.154046

        # max_time_steps == 7
        depth = 6

        # seq_len_0 == 5
        targets_0 = [0, 1, 2, 1, 0]
        loss_log_prob_0 = -3.34211
        # dimensions are time x depth
        input_prob_matrix_0 = np.asarray(
            [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
            dtype=np.float32)
        input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
        gradient_log_prob_0 = np.asarray(
            [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
            dtype=np.float32)

        # seq_len_1 == 5
        targets_1 = [0, 1, 1, 0]
        loss_log_prob_1 = -5.42262
        # dimensions are time x depth

        input_prob_matrix_1 = np.asarray(
            [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
             [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
            dtype=np.float32)
        input_log_prob_matrix_1 = np.log(input_prob_matrix_1)
        gradient_log_prob_1 = np.asarray(
            [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
             [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
            dtype=np.float32)

        # len max_time_steps array of 2 x depth matrices
        inputs = [np.vstack([input_log_prob_matrix_0[t, :],
                             input_log_prob_matrix_1[t, :]])
                  for t in range(5)] + 2 * [np.nan*np.ones((2, depth), np.float32)]

        # convert inputs into [max_time x batch_size x depth tensor] Tensor
        inputs = np.asarray(inputs, dtype=np.float32)

        # len batch_size array of label vectors
        labels, label_lengths = get_dense_array([targets_0, targets_1])

        # batch_size length vector of sequence_lengths
        input_lens = np.array([5, 5], dtype=np.int32)

        # output: batch_size length vector of negative log probabilities
        loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)

        # output: len max_time_steps array of 2 x depth matrices
        grad_truth = [np.vstack([gradient_log_prob_0[t, :],
                                 gradient_log_prob_1[t, :]])
                      for t in range(5)] + 2 * [np.zeros((2, depth), np.float32)]

        # convert grad_truth into [max_time x batch_size x depth] Tensor
        grad_truth = np.asarray(grad_truth)

        self._testCTCLoss(inputs, input_lens, labels, label_lengths,
                          loss_truth, grad_truth)


if __name__ == "__main__":
  tf.test.main()