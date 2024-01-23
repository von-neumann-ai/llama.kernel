# llama.dpcpp

LLAMA run on DPC++

# Plan

Code will be inspired by llama2.c - specifically `run.c` file here: https://github.com/karpathy/llama2.c/blob/master/run.c.

I'll be referring to lot of code from [oneapi-samples](https://github.com/oneapi-src/oneAPI-samples) to write code in DPC++. Advantage of writing code in DPC++ is that it can be run on almost any accelrator including Nvidia, AMD and Intel GPUs along with FPGAs. Besides the abstraction of buffers and heirarchical kernels make life easy.

My development environment:
1. Arc 370m laptop (4GB vRAM)
2. Ubuntu 22.04.3
3. Intel GPU Driver installed in distrobox

Later on I'll make installation of environment significantly easier by packaging everything with guix.

## Software Strategy

Use as many functions from OneMKL and OneDNN to get the first prototype out even though they might not support lot of quantization out of the box. However, know the limitations of these libraries so that quantization can be supported in the future with custom kernels.

Building blocks we need:
1. Matrix multiplcation (gemm): https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html
2. Softmax: https://oneapi-src.github.io/oneDNN/dev_guide_softmax.html
3. RMS Norm: Gotta write my own kernel

Use buffers instead of unified shared memory because they're easier to manipulate. Use high level constructs like heirarchical kernels as much as possible because of ease of use. Keep everything async and let DPC++ queue take care of kernel execution.

## Quantization Strategy

Majority of the time is in transfer of the weights from vram to controller. It doesn't really matter what representation controller is operating on as long as we're able to transfer the data from memory to controller in low precision.

In other words, we should be transfering weights data in low precision and cast them to high precision in the kernel/local memory. A way to achieve this is to pack the 8 x int4 weights into single int32 and recast them in the kernel using bit level operations. Therefore, we can see quantization as a tiling strategy.

## Packaging Strategy

One important problem for people is installation/compilation. Installation of Intel Drivers and OneAPI is not for faint of the heart. Experience is nowhere close to Nvidia drivers and that's saying something because I hate Nvidia driver experience as well.

But the best part about OneAPI is that it is completely opensource. I am hoping to use guix pacakge manager to compile OneAPI from source and package things easily.

Target experience:

```
guix install llama.dpcpp
```



## Known Bugs

1. Just had to restart the machine to get GPU working again. Likely driver crash. Restarting distrobox didn't help. Note that uptime was more than 10/15 days though. Not cool though.