#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"
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
    const Tensor& data_lens_t = context->input(1);
    const Tensor& labels_t = context->input(2);
    const Tensor& label_lens_t = context->input(3);
    const Tensor& alphabet_size_t = context->input(4);
    auto data = data_t.flat<float>();
    auto data_lens = data_lens_t.flat<int>();
    auto labels = labels_t.flat<int>();
    auto label_lens = label_lens_t.flat<int>();
    int alphabet_size = alphabet_size_t.flat<int>()(0);
    int n_minibatches = data_t.dim_size(1);

    ctcComputeInfo info;
    info.loc = CTC_CPU;
    // // TODO: num_threads with value from TF's device class
    info.num_threads = 1;

    size_t cpu_alloc_bytes;
    ctcStatus_t stat_alloc = get_workspace_size(label_lens.data(), data_lens.data(),
                                          alphabet_size, data_lens.size(), info,
                                          &cpu_alloc_bytes);

    OP_REQUIRES(context, (stat_alloc == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC memory estimation"))

    // allocate scratch space for ctc computation
    Allocator* a = cpu_allocator();
    void* scratch = a->AllocateRaw(1, cpu_alloc_bytes);

    // allocate gradient tensor
    Tensor* gradients = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, data_t.shape(),
                                                     &gradients));
    auto grads = gradients->flat<float>();

    // compute CTC
    float costs;
    ctcStatus_t stat_compute = compute_ctc_loss(data.data(),
                                                grads.data(),
                                                labels.data(),
                                                label_lens.data(),
                                                data_lens.data(),
                                                alphabet_size,
                                                n_minibatches,
                                                &costs,
                                                scratch,
                                                info);

    a->DeallocateRaw(scratch);

    OP_REQUIRES(context, (stat_compute == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC computation"))

    Tensor* loss_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), &loss_t));
    auto loss = loss_t->flat<float>();
    loss(0) = costs;

    // std::cout << costs << std::endl;
    // std::cout << stat_alloc << std::endl;
    // std::cout << stat_compute << std::endl;
    // std::cout << cpu_alloc_bytes << std::endl;
    // std::cout << n_minibatches << std::endl;
    // std::cout << info.num_threads << std::endl;
    // std::cout << alphabet_size << std::endl;

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
