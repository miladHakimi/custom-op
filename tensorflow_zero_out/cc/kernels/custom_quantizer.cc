#include "tensorflow/core/framework/op_kernel.h"
#include <string.h>
#include <cmath>

using namespace tensorflow;
class CustomQuantizerOp : public OpKernel {
    public:
    int bit_width;
    explicit CustomQuantizerOp(OpKernelConstruction* context) : OpKernel(context) {
        context->GetAttr("bit_width", &bit_width);
        // printf("bitwidth = %d \n\n\n\n", *a);
    }
    void Compute(OpKernelContext* context) override {
        
        const Tensor& input_tensor1 = context->input(0);
       
        int FIL_H = input_tensor1.shape().dim_size(0);
        int FIL_W =input_tensor1.shape().dim_size(1);
        int count = input_tensor1.shape().dim_size(2);
        int depth = input_tensor1.shape().dim_size(3);

        auto filter = input_tensor1.flat<float>();
        
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(), &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        for (int i = 0; i < count*FIL_H*FIL_W*depth; i++){
            output_flat(i) = ((int)(filter(i) * pow(2, bit_width-1) + 0.5))/(pow(2, bit_width-1)*1.0);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CustomQuantizer").Device(DEVICE_CPU), CustomQuantizerOp);