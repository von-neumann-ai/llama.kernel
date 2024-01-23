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

Decided to use Intel's libraries as opposed to my custom kernels. This means I can't go further than what these libraries offer in terms of quantization. See appendix below for detailed benchmarking.

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

# Appendix

## Design Decision on Kernels

We have two options:

1. Use libraries from OneAPI like OneDNN and OneMKL so that building blocks below are taken from them directly. These kernels are highly performant but they are not portable. In particular, kernels for intel GPUs are not even open source. This will mean I might not have flexibility to implement custom kernels.
2. Write custom kernels as much as possible even though they might not be very performant. This is because we know that maximum performance benefit is in quantization and utilization of bandwidth (see below). These custom kernels can then be optimized for quantization. For testing, compare customer kernels with serial cpp code.

Not sure which one to chose. If #2 is able to get 50% of the perf of #1, I would go with #2. Need to understand how much work it is to get there. I wish I have something like triton where I can write kernel in high level but it gets autotuned to the GPU I am using.

To resolve this, I did some benchmarking on my machine by compiling [code from here](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/GPU-Opt-Guide/libraries-kernel) (I made small modification by running kernel in loops):

```
$ ./matmul_onemkl 1024 1024
Running on: Intel(R) Arc(TM) A370M Graphics
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.105589 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.00151356 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000875378 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000880327 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000973431 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000943086 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000790954 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000811881 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000799759 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.00079287 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000809851 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000793364 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000837521 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000787656 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000835357 seconds.
oneMKL SGEMM of 1024 x 1024 and 1024 x 1024 matrices took 0.000787074 seconds.
Program completed without errors.

$ ./naive_matmul_sycl 1024 1024
Running on: Intel(R) Arc(TM) A370M Graphics
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.94135 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110893 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.111184 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.111221 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.109634 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110257 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110544 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110772 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.111249 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110385 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110823 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110827 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110138 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110781 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110267 seconds.
Naive DPC++ multiplication of 1024 x 1024 and 1024 x 1024 matrices took 0.110174 seconds.
Program completed without errors.
```

As you can see, mkl is significantly faster than fairly naive kernel! To compute flops from time to run the above 1024 square matrix multiplication, here's the formula:

```python
# insert number from above
time_per_iter = 0.110174
N = 1024
num_ops = 2 * (N ** 3)
flops = num_ops / time_per_iter
gflops = flops / 1e9
print(gflops)
```

For onemkl, we get gflops = 2706 and for naive matmul, we get gflops = 19. This means our naive kernel has 19/2706 < 1% perf of the kernel from onemkl. Therefore, we reject the approach of writing custom kernels as of now.

To confirm these benchmarks, I have also run benchmarks of [code from the dpc++ book](https://github.com/Apress/data-parallel-CPP). In this book, chapter 9 and 15 has matmul examples with different optimizations. I have modified these benchmarks to use matrix size of 1024 and ran benchmarks. Here are some numbers:

```
fig_9_4_naive_matmul
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 265.018

fig_9_8_ndrange_tiled_matmul
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 17.7596

fig_9_11_matmul_broadcast
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 17.7605

fig_9_12_ndrange_sub_group_matmul
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 189.695

fig_15_5_somewhat_parallel_matrix_multiplication
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 8.79453

fig_15_7_more_parallel_matrix_multiplication
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 265.421

fig_15_12_small_work_group_matrix_multiplication
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 15.3246

fig_15_18_columns_matrix_multiplication
Running on device: Intel(R) Arc(TM) A370M Graphics
Success!
GFlops: 32.194
```

Again these numbers confirm bad perf on custom kernels and best among these is still < 10% of onemkl kernels. So we'll stick with libraries.

## Triton

As I said above, quite envious of [performance obtained by triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) which is comparable to cublas in Nvidia GPUs. Gotta experiment with [Intel's version of triton](https://github.com/intel/intel-xpu-backend-for-triton). If it's not upto mark, I will consider writing my own compiler.

I tried installing released version of Intel triton and that [is buggy](https://github.com/intel/intel-xpu-backend-for-triton/issues/334). Build from source also [failed](https://github.com/intel/intel-xpu-backend-for-triton/issues/335). I will revisit this after a few days.

Triton seems to be pretty powerful abstraction: small code not dissimilar from hierarchical kernel of dpc++ is able to produce onemkl level of perf (on Nvidia GPUs so far though). Gotta confirm if this performance portability also works for Intel GPUs. If so, I'll work on my non-pytorch tied version of triton.