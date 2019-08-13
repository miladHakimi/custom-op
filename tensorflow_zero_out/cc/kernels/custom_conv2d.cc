#include "tensorflow/core/framework/op_kernel.h"
#include <string.h>
#include <cmath>


using namespace tensorflow;
class CustomConv2dOp : public OpKernel {
    public:
    explicit CustomConv2dOp(OpKernelConstruction* context) : OpKernel(context) {
        // context->GetAttr("bit_width", &bit_width);
    }
    void Compute(OpKernelContext* context) override {
        int bit_width = 8;        
        const Tensor& input_tensor1 = context->input(0);
        const Tensor& input_tensor2 = context->input(1);
        const Tensor& input_tensor3 = context->input(2);
        const Tensor& input_tensor4 = context->input(3);

        auto padding = input_tensor4.flat<string>();
        auto strides = input_tensor3.flat<int>();
        
        int FIL_H = input_tensor2.shape().dim_size(0);
        int FIL_W =input_tensor2.shape().dim_size(1);
        int IMG_H = input_tensor1.shape().dim_size(1);
        int IMG_W = input_tensor1.shape().dim_size(2);
        int width , height, count, depth;

        if(padding(0) == "SAME"){
            width = ceil(IMG_W/strides(1));
            height = ceil(IMG_H/strides(0));
        } 
        else{
            width = ceil((input_tensor1.shape().dim_size(2)-input_tensor2.shape().dim_size(1)+1)/strides(1));
            height = ceil((input_tensor1.shape().dim_size(1)-input_tensor2.shape().dim_size(0)+1)/strides(0));
        }

        int pad_along_height = std::max((height-1)*strides(1)+FIL_H-IMG_H, 0);
        int pad_along_width = std::max((width-1)*strides(2)+FIL_W-IMG_W, 0);
        int pad_top = pad_along_height/2;
        int pad_bottom = pad_along_height - pad_top;
        int pad_left = pad_along_width/2;
        int pad_right = pad_along_width - pad_left;

        count = input_tensor1.shape().dim_size(0);
        depth = input_tensor2.shape().dim_size(3);

        // Create an output tensor        
        tensorflow::TensorShape ts(input_tensor1.shape());
        ts.set_dim(0, count);
        ts.set_dim(1, height);
        ts.set_dim(2, width);
        ts.set_dim(3, depth);
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, ts, &output_tensor));
        auto output_flat = output_tensor->flat<float>();
        
        auto img = input_tensor1.flat<float>();
        auto filter = input_tensor2.flat<float>();
        float img_op, p_conv;
        float filter_op;
        // convolution body
        int comp_img_height = 0;
        int comp_img_width = 0;
        for (int m = 0; m < count; m++){
            for (int n = 0; n < depth; n++){
                for (int i = 0; i < height; i+=1){
                    for (int j = 0; j < width; j+=1){
                        p_conv = 0;
                        for (int k = 0; k < FIL_H; k+=1){
                            comp_img_height = i * strides(0) + k - pad_top;
                            if(comp_img_height >= 0 && comp_img_height < IMG_H + pad_bottom)
                                for (int l = 0; l < FIL_W; l+=1){
                                    comp_img_width = j * strides(0) + l - pad_left;
                                    filter_op = (int)(filter((k * FIL_W + l) * depth + n)* pow(2, bit_width-1) + 0.5);
                                    img_op =  (int)(img( m * (IMG_H*IMG_W) + comp_img_height * IMG_W + comp_img_width)* pow(2, bit_width-1) + 0.5);
                                    if(comp_img_width >= 0 && comp_img_width < IMG_W + pad_right)
                                        p_conv += ((int)(filter_op * img_op))>>(bit_width-1);
                                    else
                                        break;       
                                }    
                            else
                                break;  
                        }
                        output_flat(m*height*width*depth+(i*width+j)*depth+n) = p_conv/((pow(2, bit_width-1))*1.0);
                    }
                }
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CustomConv2d").Device(DEVICE_CPU), CustomConv2dOp);