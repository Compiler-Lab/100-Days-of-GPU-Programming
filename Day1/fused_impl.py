import torch
import torch.nn as nn

import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def fused_infer(input_ptr,output_ptr,
                weight_ptr,bias_ptr,
                input_row_stride,weight_row_stride,output_row_stride,
                M,K,N,
                BLOCK_N: tl.constexpr,BLOCK_K: tl.constexpr,
                HAS_BIAS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_n = pid_n*BLOCK_N + tl.arange(0,BLOCK_N)
    acc = tl.zeros((BLOCK_N,),dtype=tl.float32)
    for k in range(0,K,BLOCK_K):
        offset_k = k + tl.arange(0,BLOCK_K)
        x=tl.load(input_ptr + pid_m * input_row_stride + offset_k , mask=offset_k<K,other=0.0)

        w=tl.load(weight_ptr + offset_n[:,None]*weight_row_stride + offset_k[None,:],
                    mask=(offset_n[:,None]<N) & (offset_k[None,:]<K),
                    other=0.0)
        
        acc+=tl.sum(w * x[None, :], axis=1)
    
    if HAS_BIAS:                                   
        bias = tl.load(bias_ptr + offset_n, mask=offset_n < N, other=0.0)
        acc += bias
        
    tl.store(output_ptr + pid_m*output_row_stride + offset_n ,acc,mask=offset_n<N)
    

# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
        W = model.weight.detach().contiguous()
        has_bias = model.bias is not None
        b = model.bias.detach().contiguous() if has_bias else W  
        M,K=input.shape #input[M,K] M[N,K] output[M,N]
        N=W.shape[0] 
        BLOCK_N=128
        BLOCK_K=128
        grid=(M,triton.cdiv(N,BLOCK_N))
        fused_infer[grid](input,output,W,b,
                    input.stride(0),W.stride(0),output.stride(0),
                    M,K,N,
                    BLOCK_N=BLOCK_N,BLOCK_K=BLOCK_K,HAS_BIAS=has_bias)

#0.26ms in A100