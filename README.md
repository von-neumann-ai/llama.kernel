# llama.dpcpp

LLAMA run on DPC++

# Plan

Code will be inspired by llama2.c - specifically `run.c` file here: https://github.com/karpathy/llama2.c/blob/master/run.c.

I read book [Data Parallel C++](https://link.springer.com/book/10.1007/978-1-4842-5574-2) to understand the programming model of SYCL. The book is very easy to pick up and understand. It's free to download BTW. The book comes with [code samples](https://github.com/Apress/data-parallel-CPP) as well.


I'll be referring to lot of code from [oneapi-samples](https://github.com/oneapi-src/oneAPI-samples) to write code in DPC++. Advantage of writing code in DPC++ is that it can be run on almost any accelrator including Nvidia, AMD and Intel GPUs along with FPGAs. Besides the abstraction of buffers and heirarchical kernels make life easy.

My development environment:
1. Arc 370m laptop (4GB vRAM)
2. Ubuntu 22.04.3
3. Intel GPU Driver installed in distrobox

Later on I'll make installation of environment significantly easier by packaging everything with guix.

## Software Strategy

Design decision pending: We have two options:

1. Use libraries from OneAPI like OneDNN and OneMKL so that building blocks below are taken from them directly. These kernels are highly performant but they are not portable. In particular, kernels for intel GPUs are not even opensource. This will mean I might not have flexibility to implement custom kernels.
2. Write custom kernels as much as possible even though they might not be very performant. This is because we know that maximum performance benefit is in quantization and utilization of bandwidth (see below). These custom kernels can then be optimized for quantization. For testing, compare customer kernels with serial cpp code.

Not sure which one to chose. If #2 is able to get 50% of the perf of #1, I would go with #2. Need to understand how much work it is to get there.

I wish I have something like triton where I can write kernel in high level but it gets autotuned to the GPU I am using.

Building blocks we need:

1. Matrix multiplcation
2. Softmax
3. RMS Norm

Use buffers instead of unified shared memory because they're easier to manipulate. Use high level constructs like heirarchical kernels as much as possible because of ease of use. Keep everything async and let DPC++ queue take care of kernel execution.

## Performance Strategy

Majority of the time is in transfer of the weights from vram to controller. It doesn't really matter what representation controller is operating on as long as we're able to transfer the data from memory to controller in low precision.

In other words, we should be transfering weights data in low precision and cast them to high precision in the kernel/local memory. A way to achieve this is to pack the 8 x int4 weights into single int32 and recast them in the kernel using bit level operations. Therefore, we can see quantization as a tiling strategy.

Advertized TFLOPs/TOPs numbers of a chip don't matter beyond a point for LLM inference because those numbers can only be achieved when the weights are in the cache. But in LLM inference, model weights don't fit in the cache and most of the time goes into transferring the weights to the cache/registers. For [modern GPUs](https://docs.google.com/spreadsheets/d/1vsJUpIZdFwIYrEfCGWcGOQZhMYE_NG70qSisA2OO-Og/edit#gid=330553096), flops are around 50 times the bandwidth. This means that there's no point focussing on writing most optimal kernels in terms of compute efficiency. What I need to focus is on weight/data transfer from device's main memory.

## Packaging Strategy

One important problem for people is installation/compilation. Installation of Intel Drivers and OneAPI is not for faint of the heart. Experience is nowhere close to Nvidia drivers and that's saying something because I hate Nvidia driver experience as well.

But the best part about OneAPI is that it is completely opensource. I am hoping to use guix pacakge manager to compile OneAPI from source and package things easily.

Target experience:

```
guix install llama.dpcpp
```


## Known Bugs

1. Just had to restart the machine to get GPU working again. Likely driver crash. Restarting distrobox didn't help. Note that uptime was more than 10/15 days though. Not cool though.

## References:

Beyond the links from above, here are some assorted links:

1. https://siboehm.com/articles/22/CUDA-MMM
2. https://github.com/codeplaysoftware/portBLAS