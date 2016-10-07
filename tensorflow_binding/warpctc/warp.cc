#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"
#include "./ctc.h"
#include <iostream>


// forward declare CUstream to avoid pulling in cuda headers
typedef struct CUstream_st* CUstream;

// forward declare stream headers to avoid pulling in stream_executor code
namespace perftools {
namespace gputools {
class Stream;
namespace cuda {
  CUstream AsCUDAStreamValue(Stream *stream);
}  // namespace cuda
}  // namespace gputools
}  // namespace perftools


REGISTER_OP("WarpCTC")
    .Input("data: float32")
    .Input("data_lengths: int32")
    .Input("flat_labels: int32")
    .Input("label_lengths: int32")
    .Attr("alphabet_size: int")
    .Output("loss: float32")
    .Output("gradient: float32");

using namespace tensorflow;

class WarpCTCOpCPU : public OpKernel {
 public:
  explicit WarpCTCOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("alphabet_size", &alphabet_size_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& data_t = context->input(0);
    const Tensor& data_lens_t = context->input(1);
    const Tensor& labels_t = context->input(2);
    const Tensor& label_lens_t = context->input(3);
    auto data = data_t.flat<float>();
    auto data_lens = data_lens_t.flat<int>();
    auto labels = labels_t.flat<int>();
    auto label_lens = label_lens_t.flat<int>();
    int alphabet_size = alphabet_size_;
    int n_minibatches = data_t.dim_size(1);

    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = context->device()->tensorflow_cpu_worker_threads()->num_threads;

    size_t cpu_alloc_bytes;
    ctcStatus_t stat_alloc = get_workspace_size(label_lens.data(), data_lens.data(),
                                          alphabet_size, n_minibatches, info,
                                          &cpu_alloc_bytes);

    OP_REQUIRES(context, (stat_alloc == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC memory estimation"))

    // allocate scratch space for ctc computation
    Allocator* a = context->device()->GetAllocator(AllocatorAttributes());

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

  }
 private:
  int alphabet_size_;
};

#define EIGEN_USE_GPU

class WarpCTCOpGPU : public OpKernel {
 public:
  explicit WarpCTCOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("alphabet_size", &alphabet_size_));
  }

  void Compute(OpKernelContext* context) override {

    const Tensor& data_t = context->input(0);
    const Tensor& data_lens_t = context->input(1);
    const Tensor& labels_t = context->input(2);
    const Tensor& label_lens_t = context->input(3);
    auto data = data_t.flat<float>();
    auto data_lens = data_lens_t.flat<int32>();
    auto labels = labels_t.flat<int32>();
    auto label_lens = label_lens_t.flat<int32>();
    int alphabet_size = alphabet_size_;
    int n_minibatches = data_t.dim_size(1);

    // for debugging -- these values appear to not be set. Why?
    // the values for some reason do not appear to live on the GPU
    // maybe it is using the cpu eigen, or something?
    // float n0 = data(0);
    // int n1 = data_lens(0);
    // int n2 = labels(0);
    // int n3 = label_lens(0);

    // ctcComputeInfo info;
    // info.loc = CTC_GPU;
    // info.stream = perftools::gputools::cuda::AsCUDAStreamValue(
    //   context->device()->tensorflow_gpu_device_info()->stream);
    // size_t gpu_alloc_size;
    // ctcStatus_t stat_alloc = get_workspace_size(label_lens.data(), data_lens.data(),
    //                                             alphabet_size, n_minibatches, info,
    //                                             &gpu_alloc_size);
    // OP_REQUIRES(context, (stat_alloc == CTC_STATUS_SUCCESS),
    //             errors::Internal("Error in CTC memory estimation"))

    // // // allocate scratch space for ctc computation
    // Allocator* a = context->device()->GetAllocator(AllocatorAttributes());
    // void* scratch = a->AllocateRaw(1, gpu_alloc_size);

    // // // allocate gradient tensor
    // Tensor* gradients = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(1, data_t.shape(),
    //                                                  &gradients));
    // auto grads = gradients->flat<float>();

    // // // compute CTC
    // float costs;
    // ctcStatus_t stat_compute = compute_ctc_loss(data.data(),
    //                                             grads.data(),
    //                                             labels.data(),
    //                                             label_lens.data(),
    //                                             data_lens.data(),
    //                                             alphabet_size,
    //                                             n_minibatches,
    //                                             &costs,
    //                                             scratch,
    //                                             info);

    // a->DeallocateRaw(scratch);

    // OP_REQUIRES(context, (stat_compute == CTC_STATUS_SUCCESS),
    //             errors::Internal("Error in CTC computation"))

    // Tensor* loss_t = nullptr;
    // OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), &loss_t));
    // auto loss = loss_t->flat<float>();
    // loss(0) = 0.5;
  }
 private:
  int alphabet_size_;
};

#undef EIGEN_USE_GPU

REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(DEVICE_CPU), WarpCTCOpCPU);
REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(DEVICE_GPU), WarpCTCOpGPU);
