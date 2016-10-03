#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "../../include/ctc.h"


REGISTER_OP("CTC")
    .Input("inputs: float32")
    .Input("input_lens: int32")
    .Input("labels: int32")
    .Input("label_lens: int32")
    .Output("loss: float32");

using namespace tensorflow;

class CTCOp : public OpKernel {
 public:
  explicit CTCOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("CTC").Device(DEVICE_CPU), CTCOp);
