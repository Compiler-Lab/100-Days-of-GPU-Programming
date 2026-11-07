**Day 0 : Hello GPU kernels world :**

Backgrond: Inspired by the 100/365-day GPU programming challenges from [levi](https://x.com/levidiamode), [Umar Jamil](https://x.com/hkproj), and [sadernoheart](https://x.com/sadernoheart), I have decided to embark on my own 100-day challenge.

Although I have some foundational knowledge of topics like CUDA programming and parallel computing, it is only superficial. Therefore, I intend to use this challenge to learn GPU programming systematically—not limited strictly to CUDA; I may also explore ai-compiler-related concepts.
So stay tuned!

---

**Day 1 : Fused Linear Inference**

Implemented `nn.Linear` inference in PyTorch and Triton.
- Naive PyTorch (`model(input)`): 0.54ms on A100
- Fused Triton kernel (matmul + bias): 0.26ms on A100

Resources: [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

---

**Day 2 : Sigmoid Linear Unit (SiLU)**

Implemented SiLU (`x * sigmoid(x)`) in CUDA.
- Profiling with Nsight Compute (`ncu`) to identify the kernel bottleneck and measure memory bandwidth utilization — WIP.
