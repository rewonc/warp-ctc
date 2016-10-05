import tensorflow as tf
import numpy as np
from warpctc import ctc


def get_dense_array(labels):
    '''Return a flattened set of labels and lengths
    for use in CTC calculations

    Inputs:
        labels (list of lists): 0-indexed character labels

    Outputs:
        0 (np.array): flattened labels
        1 (np.array): label lengths
    '''
    flat = sum(labels)
    lens = [len(li) for li in labels]
    return np.array(flat, dtype=np.int32), np.array(lens, dtype=np.int32)


class CTCLossTest(tf.test.TestCase):

    def _run_ctc(self, data, data_lengths,
                 flat_labels, label_lengths,
                 alphabet_size, expected_loss,
                 expected_gradients, expected_error=None):
        self.assertEquals(data.shape, expected_gradients.shape)

        data_t = tf.constant(data)
        data_lengths_t = tf.constant(data_lengths)
        flat_labels_t = tf.constant(flat_labels)
        label_lengths_t = tf.constant(label_lengths)

        with self.test_session(use_gpu=False) as sess:
            loss = ctc(data_t, data_lengths=data_lengths_t,
                       flat_labels=flat_labels_t,
                       label_lengths=label_lengths_t,
                       alphabet_size=alphabet_size)

            grad = tf.gradients(loss, [data_t])[0]

            import pdb; pdb.set_trace()
            self.assertShapeEqual(expected_loss, loss)
            self.assertShapeEqual(expected_gradients, grad)

            if expected_error is None:
                (tf_loss, tf_grad) = sess.run([loss, grad])
                self.assertAllClose(tf_loss, expected_loss, atol=1e-6)
                self.assertAllClose(tf_grad, expected_gradients, atol=1e-6)
            else:
                with self.assertRaisesOpError(expected_error):
                    sess.run([loss, grad])

    def test_basic(self):
        # Softmax activations for the following inputs:
        # np.array([
        #     [0.1, 0.6, 0.1, 0.1, 0.1],
        #     [0.1, 0.1, 0.6, 0.1, 0.1]
        # ], dtype=np.float32)

        alphabet_size = 5
        softmax_activations = np.asarray([
            [0.177031, 0.291875, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, 0.291875, 0.177031, 0.177031]
        ], dtype=np.float32)
        # dimensions should be t, n, p: (t timesteps, n minibatches,
        # p prob of each alphabet). This is one instance, so expand
        # dimensions in the middle
        data = np.expand_dims(softmax_activations, 1)
        labels = np.asarray([1, 2], dtype=np.int32)
        expected_loss = np.asarray([0.0851911], dtype=np.float32)
        gradients = np.asarray([
            [0.177031, -0.708125, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, -0.708125, 0.177031, 0.177031]
        ])
        expected_gradients = np.expand_dims(gradients, 1)
        label_lengths = np.asarray([2], dtype=np.int32)
        data_lengths = np.asarray([2], dtype=np.int32)

        self._run_ctc(data, data_lengths=data_lengths,
                      flat_labels=labels, label_lengths=label_lengths,
                      alphabet_size=alphabet_size,
                      expected_loss=expected_loss,
                      expected_gradients=expected_gradients)

    # def _testCTCLoss(self, inputs, input_lens, labels, label_lens,
    #                  loss_truth, grad_truth, expected_err_re=None):
    #     self.assertEquals(inputs.shape, grad_truth.shape)

    #     inputs_t = tf.constant(inputs)

    #     with self.test_session(use_gpu=False) as sess:
    #         loss = ctc(inputs=inputs_t, input_lens=input_lens,
    #                    labels=labels, label_lens=label_lens)
    #         grad = tf.gradients(loss, [inputs_t])[0]

    #         self.assertShapeEqual(loss_truth, loss)
    #         self.assertShapeEqual(grad_truth, grad)

    #         if expected_err_re is None:
    #             (tf_loss, tf_grad) = sess.run([loss, grad])
    #             self.assertAllClose(tf_loss, loss_truth, atol=1e-6)
    #             self.assertAllClose(tf_grad, grad_truth, atol=1e-6)
    #         else:
    #             with self.assertRaisesOpError(expected_err_re):
    #                 sess.run([loss, grad])

    # def testBasic(self):
    #     """Test two batch entries from tensorflow tests"""

    #     depth = 6
    #     targets_0 = [0, 1, 2, 1, 0]
    #     loss_log_prob_0 = -3.34211
    #     # dimensions are time x depth
    #     input_prob_matrix_0 = np.asarray(
    #         [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
    #          [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
    #          [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
    #          [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
    #          [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
    #         dtype=np.float32)
    #     input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
    #     gradient_log_prob_0 = np.asarray(
    #         [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
    #          [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
    #          [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
    #          [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
    #          [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
    #         dtype=np.float32)

    #     targets_1 = [0, 1, 1, 0]
    #     loss_log_prob_1 = -5.42262

    #     input_prob_matrix_1 = np.asarray(
    #         [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
    #          [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
    #          [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
    #          [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
    #          [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
    #         dtype=np.float32)
    #     input_log_prob_matrix_1 = np.log(input_prob_matrix_1)
    #     gradient_log_prob_1 = np.asarray(
    #         [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
    #          [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
    #          [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
    #          [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
    #          [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
    #         dtype=np.float32)

    #     inputs = [np.vstack([input_log_prob_matrix_0[t, :],
    #                          input_log_prob_matrix_1[t, :]])
    #               for t in range(5)] + 2 * [np.nan*np.ones((2, depth), np.float32)]

    #     inputs = np.asarray(inputs, dtype=np.float32)
    #     labels, label_lengths = get_dense_array([targets_0, targets_1])
    #     input_lens = np.array([5, 5], dtype=np.int32)
    #     loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)
    #     grad_truth = [np.vstack([gradient_log_prob_0[t, :],
    #                              gradient_log_prob_1[t, :]])
    #                   for t in range(5)] + 2 * [np.zeros((2, depth), np.float32)]
    #     grad_truth = np.asarray(grad_truth)

    #     self._testCTCLoss(inputs, input_lens, labels, label_lengths,
    #                       loss_truth, grad_truth)


if __name__ == "__main__":
    tf.test.main()
