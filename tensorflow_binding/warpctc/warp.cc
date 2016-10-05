#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "./ctc.h"
#include <iostream>

REGISTER_OP("WarpCTC")
    .Input("data: float32")
    .Input("data_lengths: int32")
    .Input("flat_labels: int32")
    .Input("label_lengths: int32")
    .Input("alphabet_size: int32")
    .Output("loss: float32")
    .Output("gradient: float32");


using namespace tensorflow;

class WarpCTCOpCPU : public OpKernel {
 public:
  explicit WarpCTCOpCPU(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& data_t = context->input(0);
    // const Tensor& data_lens_t = context->input(1);
    // const Tensor& labels_t = context->input(2);
    // const Tensor& label_lens_t = context->input(3);
    // const Tensor& alphabet_size_t = context->input(4);
    auto data = data_t.flat<float>();
    // auto data_lens = data_lens_t.flat<int>();
    // auto labels = labels_t.flat<int>();
    // auto label_lens = label_lens_t.flat<int>();
    // int alphabet_size = alphabet_size_t.vec<int>()({0});

    ctcComputeInfo info;
    info.loc = CTC_CPU;
    // // Use single thread for now
    info.num_threads = 1;

    // ctcStatus_t get_workspace_size(const int* const label_lengths,
    //                            const int* const input_lengths,
    //                            int alphabet_size, int minibatch,
    //                            ctcComputeInfo info,
    //                            size_t* size_bytes);

    // size_t cpu_alloc_bytes;
    // ctcStatus_t stat = get_workspace_size(label_lens.data(), data_lens.data(),
    //                                       alphabet_size, data_lens.size(), info,
    //                                       &cpu_alloc_bytes);

    // OP_REQUIRES(context, (stat == CTC_STATUS_SUCCESS),
    //             errors::Internal("Error in CTC memory allocation"))

    // std::cout << std::to_string(cpu_alloc_bytes) << std::endl;
    std::cout << info.num_threads << std::endl;

    // calculate the required scratch size
    // Status allocate_temp(DataType type, const TensorShape& shape,
    //                      Tensor* out_temp) {
    //   return allocate_temp(type, shape, out_temp, AllocatorAttributes());
    // }

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, data_t.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Create an gradients tensor
    Tensor* gradients = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, data_t.shape(),
                                                     &gradients));
    auto grads = gradients->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = data.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
      grads(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = data(0);
  }
};

// class WarpCTCOpGPU : public OpKernel {
//  public:
//   explicit WarpCTCOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

//   void Compute(OpKernelContext* context) override {
//     // Grab the input tensors
//     const Tensor& data_t = context->input(0);
//     const Tensor& data_lens_t = context->input(1);
//     const Tensor& labels_t = context->input(2);
//     const Tensor& label_lens_t = context->input(3);
//     const Tensor& alphabet_size_t = context->input(4);
//     auto data = data_t.flat<float>();
//     auto data_lens = data_lens_t.flat<float>();
//     auto labels = labels_t.flat<float>();
//     auto label_lens = label_lens_t.flat<float>();
//     auto alphabet_size = alphabet_size_t.flat<float>();


//     // calculate the required scratch size
//     // Status allocate_temp(DataType type, const TensorShape& shape,
//     //                      Tensor* out_temp) {
//     //   return allocate_temp(type, shape, out_temp, AllocatorAttributes());
//     // }

//     // Create an output tensor
//     Tensor* output_tensor = NULL;
//     OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
//                                                      &output_tensor));
//     auto output = output_tensor->flat<float>();

//     // Create an gradients tensor
//     Tensor* gradients = NULL;
//     OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor.shape(),
//                                                      &gradients));
//     auto grads = gradients->flat<float>();

//     // Set all but the first element of the output tensor to 0.
//     const int N = input.size();
//     for (int i = 1; i < N; i++) {
//       output(i) = 0;
//       grads(i) = 0;
//     }

//     // Preserve the first input value if possible.
//     if (N > 0) output(0) = input(0);
//   }
// };

REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(DEVICE_CPU), WarpCTCOpCPU);
// REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(DEVICE_GPU), WarpCTCOpGPU);
