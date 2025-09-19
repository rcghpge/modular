# Contents

- [AMD Print Lessons Learned](amd-printf-lessons-learned.md)

    This document describes the technical challenges and solutions
    involved in implementing print statement debugging for AMD GPU
    kernels in Mojo by porting OpenCL hostcall functionality to avoid
    dependencies on AMD's device-libs and additional LLVM copies.

- [FP8 Support in Mojo](fp8-support-in-mojo.md)

    This document describes the implementation of FP8 floating point
    support in the Mojo programming language, including plumbing the
    new data types through the stack and developing math approximations
    optimized for the reduced precision format.

- [Element-wise Operations on GPUs](elementwise-ops.md)

    This document analyzes element-wise GPU operations performance across
    different NVIDIA accelerators (A100, A10, L4), examining memory bandwidth
    limitations, cache effects, vectorization strategies, and implementation
    optimizations for operations like memcpy in CUDA and Mojo.

- [GenAI and Paged Attention](genai-paged-attention.md)

    This document explains PagedAttention, a memory management
    technique for LLM inference that fragments KV cache into reusable
    pages and enables prefix sharing between sequences with common
    prompts, resulting in improved memory efficiency and faster
    time-to-first-token performance.

- [Matrix Multiplication on Blackwell Part
  1—Introduction](matmul-on-blackwell-part-1.md)

    This document introduces matrix multiplication fundamentals for LLMs,
    explains GPU architecture evolution from Ampere through Blackwell, and
    implements a basic 4-line matmul kernel achieving 5 TFLOPs performance.

- [Matrix Multiplication on Blackwell: Part 2—Using Hardware Features to
  Optimize Matmul](matmul-on-blackwell-part-2.md)
    This document demonstrates optimization techniques including TMA async
    loading, tensor cores, shared memory tiling, and swizzling to achieve 58x
    performance improvement over the naive implementation, reaching 293 TFLOPs.

- [Matrix Multiplication on Blackwell: Part 3-The Optimizations Behind 85% of
  SOTA Performance](matmul-on-blackwell-part-3.md)

    This document implements advanced optimizations using 2SM MMA instructions,
    CTA memory multicasting, warp specialization, and pipelined
    double-buffering to achieve 85% of state-of-the-art performance at 1,493
    TFLOPs.

- [Matrix Multiplication to Flash Attention](matmul-to-flash-attention.md)

    This document explains how Flash Attention can be understood as an extension
    of fast matrix multiplication techniques for Ampere hardware, using
    asynchronous data transfer instructions and online softmax computation to
    achieve memory-efficient attention processing without materializing large
    intermediate matrices.

- [Multi-Head Flash Attention](multi-head-flash-attention.md)

    This document describes the implementation of multi-head attention using
    Flash Attention algorithms, progressing from the basic self-attention
    mechanism, through Flash Attention 2's memory-efficient tiling approach, to
    Flash Attention 3's specialized optimizations for Hopper architecture with
    asynchronous operations and warp-group specialization.

- [U/WGMMA Flash Decoding](uwgmma-flash-decoding.md)

    This document explores "U/WGMMA Flash Decoding," proposing to transpose matrix
    operations in Flash Attention 3 to better utilize GPU hardware by operating
    on 64+ rows at once instead of wasting computation on smaller group sizes,
    while analyzing the trade-offs between improved throughput and increased
    memory/synchronization costs.

- [Multi-Head Latent Attention](multi-head-latent-attention.md)

    This design document presents an optimized Multi-head Latent
    Attention (MLA) implementation that reduces KV cache memory usage
    to just 576 values per token by storing compressed latent
    representations instead of full K and V tensors.

- [PyTorch Layers to MAX Mapping Guide](pytorch-to-max-mapping-guide.md)

    This guide provides mappings between common PyTorch layers used in
    HuggingFace `transformers` and their equivalent MAX graph operations and
    layer abstractions.

- [Token sampling](token-sampling.md)

    This design document provides a comprehensive overview of token
    sampling techniques in LLM inference, covering algorithms like
    greedy sampling, top-k, top-p, and min-p sampling that control the
    randomness and diversity of text generation by determining how the
    next token is selected from probability distributions.

- [WGMMA Programming](wgmma-programming.md)

    This document explains the WGMMA (Warp Group MMA) tensor core instruction
    introduced in Hopper H100 GPUs, covering its advantages over regular MMA
    instructions, memory layout requirements for matrices in shared memory, and
    providing a complete CUDA implementation example.
