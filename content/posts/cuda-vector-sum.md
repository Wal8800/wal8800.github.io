---
title: "Vector sum"
date:   2026-04-28 06:00:48 +1300
---

### The problem

The challenge is to reduce a vector of `float32` values into a single sum.

<figure>
  <svg viewBox="0 0 660 300" width="660" height="300" role="img" aria-labelledby="vector-sum-title vector-sum-desc" style="width: 100%; max-width: 660px; height: auto;">
    <title id="vector-sum-title">Small float32 vector reduction</title>
    <desc id="vector-sum-desc">Eight contiguous float32 values are read from an input vector and summed into one output value.</desc>
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#4b5563" />
      </marker>
    </defs>
    <text x="24" y="34" fill="#111827" font-family="monospace" font-size="18">input vector: float32 x[8]</text>
    <text x="408" y="258" fill="#111827" font-family="monospace" font-size="18">output</text>
    <g font-family="monospace" font-size="14" text-anchor="middle">
      <g transform="translate(24, 64)">
        <rect width="74" height="58" rx="6" fill="#dbeafe" stroke="#2563eb" />
        <text x="37" y="25" fill="#111827">x[0]</text>
        <text x="37" y="45" fill="#1f2937">1.0</text>
        <text x="37" y="82" fill="#6b7280">byte 0</text>
      </g>
      <g transform="translate(98, 64)">
        <rect width="74" height="58" rx="6" fill="#dbeafe" stroke="#2563eb" />
        <text x="37" y="25" fill="#111827">x[1]</text>
        <text x="37" y="45" fill="#1f2937">2.0</text>
        <text x="37" y="82" fill="#6b7280">byte 4</text>
      </g>
      <g transform="translate(172, 64)">
        <rect width="74" height="58" rx="6" fill="#dbeafe" stroke="#2563eb" />
        <text x="37" y="25" fill="#111827">x[2]</text>
        <text x="37" y="45" fill="#1f2937">-1.5</text>
        <text x="37" y="82" fill="#6b7280">byte 8</text>
      </g>
      <g transform="translate(246, 64)">
        <rect width="74" height="58" rx="6" fill="#dbeafe" stroke="#2563eb" />
        <text x="37" y="25" fill="#111827">x[3]</text>
        <text x="37" y="45" fill="#1f2937">4.0</text>
        <text x="37" y="82" fill="#6b7280">byte 12</text>
      </g>
      <g transform="translate(320, 64)">
        <rect width="74" height="58" rx="6" fill="#dcfce7" stroke="#16a34a" />
        <text x="37" y="25" fill="#111827">x[4]</text>
        <text x="37" y="45" fill="#1f2937">0.5</text>
        <text x="37" y="82" fill="#6b7280">byte 16</text>
      </g>
      <g transform="translate(394, 64)">
        <rect width="74" height="58" rx="6" fill="#dcfce7" stroke="#16a34a" />
        <text x="37" y="25" fill="#111827">x[5]</text>
        <text x="37" y="45" fill="#1f2937">3.0</text>
        <text x="37" y="82" fill="#6b7280">byte 20</text>
      </g>
      <g transform="translate(468, 64)">
        <rect width="74" height="58" rx="6" fill="#dcfce7" stroke="#16a34a" />
        <text x="37" y="25" fill="#111827">x[6]</text>
        <text x="37" y="45" fill="#1f2937">-2.0</text>
        <text x="37" y="82" fill="#6b7280">byte 24</text>
      </g>
      <g transform="translate(542, 64)">
        <rect width="74" height="58" rx="6" fill="#dcfce7" stroke="#16a34a" />
        <text x="37" y="25" fill="#111827">x[7]</text>
        <text x="37" y="45" fill="#1f2937">1.5</text>
        <text x="37" y="82" fill="#6b7280">byte 28</text>
      </g>
    </g>
    <path d="M 330 158 L 330 214" fill="none" stroke="#4b5563" stroke-width="2.5" marker-end="url(#arrowhead)" />
    <g transform="translate(272, 218)" font-family="monospace" text-anchor="middle">
      <rect width="116" height="70" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="2" />
      <text x="58" y="30" fill="#111827" font-size="16">sum</text>
      <text x="58" y="54" fill="#1f2937" font-size="18">8.5</text>
    </g>
  </svg>
</figure>

I'm running the benchmark test case to evaluate the performance of the kernels.

The timings are generated from a version of this [eval script](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp_v2/eval.py) but with minor changes to run on Modal.


### Base Submission

The [base submission](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp_v2/vectorsum_py/submission.py) benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.116 |
| 3,276,800 | 0.104 |
| 6,553,600 | 0.107 |
| 13,107,200 | 0.104 |
| 26,214,400 | 0.123 |
| 52,428,800 | 0.177 |

The base submission is a Triton kernel that splits the input vector into 1024-element CTA blocks.
Each block reads the data, performs a local sum, then atomically adds the local sum to compute the final output.

### PMPP baseline

The Programming Massively Parallel Processors book walks through a solution that provides:

**Global memory coalescing**

The threads in the CTA are reading values adjacent to each other. This means their global memory read requests are merged into one, allowing more requests and increasing memory traffic bandwidth.

**Thread coarsening**

Each thread is doing more computation by reading more input values in the initial pass and summing them. The aim is to provide better latency hiding, while some blocks are reading global memory, there is a bigger overlap for some other blocks doing computation.


```cpp

constexpr int BLOCK_DIM = 1024;
constexpr int COARSE_FACTOR = 2;

__global__ void CoarsenedSumReductionKernel(
  float* __restrict__ input,
  float* __restrict__ output,
  int n_elements
) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;
  float sum = input[i];
  for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
    unsigned int idx = i + tile * BLOCK_DIM;
    if (idx < n_elements) {
      sum += input[idx];
    }
  }

  input_s[t] = sum;

  for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}
```

PMPP solution benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.026 |
| 3,276,800 | 0.026 |
| 6,553,600 | 0.032 |
| 13,107,200 | 0.051 |
| 26,214,400 | 0.075 |
| 52,428,800 | 0.125 |

With the global memory coalescing and thread coarsening, it performs better than the base submissions. This is especially true for smaller test cases, where the overhead matters more.

### CUB baseline

CUB is a library provided by NVIDIA. It contains a function `cub::DeviceReduce::Sum` to perform sum reduction.

```cpp
void run_vectorsum_cuda(torch::Tensor x, torch::Tensor out, torch::Tensor temp_storage) {
  size_t temp_storage_bytes = static_cast<size_t>(temp_storage.numel());

  // C10_CUDA_CHECK is PyTorch’s CUDA error wrapper.
  C10_CUDA_CHECK(cub::DeviceReduce::Sum(
      static_cast<void*>(temp_storage.data_ptr<uint8_t>()),
      temp_storage_bytes,
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      num_items(x)));
}
```

CUB baseline benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.012 |
| 3,276,800 | 0.016 |
| 6,553,600 | 0.021 |
| 13,107,200 | 0.032 |
| 26,214,400 | 0.052 |
| 52,428,800 | 0.085 |

As expected from the official NVIDIA library, it performs very well out of the box.

### CuteDSL baseline (v2)

I used Codex to generate an initial implementation in [CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html). The solution consists of two kernels:

1. An initial kernel that processes the input vector
2. Then a smaller kernel that launches a single CTA (256 threads) and reduces the partials buffer into a single sum.

CuteDSL baseline benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.034 |
| 3,276,800 | 0.038 |
| 6,553,600 | 0.034 |
| 13,107,200 | 0.046 |
| 26,214,400 | 0.069 |
| 52,428,800 | 0.106 |


This is better than the base submission and PMPP baseline but still far away from the CUB baseline.

### v3

Key changes:
- Switch to CUDA C++ because I couldn't figure out how to express vectorised loads in CuteDSL so that it results in a vectorised load PTX instruction.
- Each block processed a single tile compared to multiple tiles:
  - Through empirical experiments, it looks like this provides better latency hiding.
  - Something about having the stride loop within a single block made it slightly slower
- Remove the second reduce kernel and perform an atomic add
  - the second kernel adds relatively significant overhead for smaller test cases

Benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.024 |
| 3,276,800 | 0.024 |
| 6,553,600 | 0.025 |
| 13,107,200 | 0.036 |
| 26,214,400 | 0.058 |
| 52,428,800 | 0.094 |


### v4

The kernel logic is largely the same but I noticed with the ncu profile, the single kernel runtime was shorter than the mean time taken from the modal eval script.
This led me to realise there were some non-kernel overheads so I proceeded to do some cleanups. They are, in order of impact:

- Avoiding PyTorch dispatch and validation work. In v3, I naively called `output.zero_()` before running the kernel. I didn't realise this actually launches a kernel to set the value to 0 in CUDA. In v4, we call this line `cudaMemsetAsync(out.data_ptr<float>(), 0, sizeof(float));` to set the output to 0 before running the kernel.

- Remove grid-stride loop logic. The parameter already only processes 1 tile per block, so the loop logic wasn't needed.
- Codex was too defensive even though the inputs are fixed shape and types:
  - Remove logic to handle unaligned data (not 16-byte aligned). For the problem, the input is always 16-byte aligned.
  - Various checks on the device, input and output shape and dtypes. Those are not needed since the problem inputs are fixed.


Benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.011 |
| 3,276,800 | 0.014 |
| 6,553,600 | 0.019 |
| 13,107,200 | 0.030 |
| 26,214,400 | 0.049 |
| 52,428,800 | 0.082 |

### v5

v4 does float4 global loads

```cpp
float4 v = x4[vec_index];
acc += v.x + v.y + v.z + v.w;
v = x4[vec_index + kThreadsPerBlock];
```

v5 uses __ldcs for those float4 loads:

```cpp
float4 v0 = __ldcs(&x4[vec_index]);
float4 v1 = __ldcs(&x4[vec_index + kThreadsPerBlock]);
```

`__ldcs` translates to PTX instructions that load data and provide a streaming cache-policy hint for the data read. This is a better fit for the current problem as the data are only read **once** for the sum reduction.

Benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.011 |
| 3,276,800 | 0.013 |
| 6,553,600 | 0.018 |
| 13,107,200 | 0.027 |
| 26,214,400 | 0.045 |
| 52,428,800 | 0.080 |

### Accumulated errors with large float32 vectors

While v5 got faster, I started encountering errors when submitting to the leaderboard.

The submitted result was greater than the allowed error threshold when compared with the reference kernel.

This is when I realised that adding up a large list of `float32` values does not always produce the same result.

**When two `float32` values are added together, the exact mathematical result may not be representable as another `float32`. The result is rounded to the nearest representable `float32` value.**

Because of this rounding, `float32` addition is not associative. For example (_generated with codex_):

```python
import numpy as np

a = np.float32(1e8)
b = np.float32(-1e8)
c = np.float32(1.0)

x = np.float32(np.float32(a + b) + c)
y = np.float32(a + np.float32(b + c))

print("(a + b) + c =", x)
print("a + (b + c) =", y)
```

```console
(a + b) + c = 1.0
a + (b + c) = 0.0
```

The difference comes from the intermediate sums. Each addition rounds its result back to float32, so changing the order of the additions changes where rounding happens.

With only a few values, this difference is usually tiny. But when reducing a very large vector, those small differences can accumulate.

In test mode, a vector size of 2,048 was still well within the error threshold. However, with a vector size of 52,428,800 in benchmark mode, the error became large enough to fail the leaderboard check.

The reference kernel mitigated this by converting the values to float64 during accumulation. This improves the precision of the intermediate sums, so less rounding error accumulates before the final result is produced.

Running test mode with the large vector size, it looks like baselines and previous versions have relatively lower errors.

My speculation is that this comes from relatively fewer unordered atomic adds compared to the v5 kernel.

### v6

With the above in mind, I'm aiming to maintain/improve the latency while keeping the accumulated errors so I can submit to the leaderboard.

Key changes:

- Reduce the number of CTAs, reducing the number of atomic adds
  - v6 uses 32768 floats/CTA, so it launches 1600 CTAs
  - v5 uses 8192 floats/CTA, so it launches 6400 CTAs.
- With the increased float computation per CTA, we try to maintain latency by:
  - Reduce `__launch_bounds__(256, 8)` to `__launch_bounds__(256, 4)`
    - Relaxing the compiler hint from at least 8 CTA blocks per SM to 4 CTA blocks per SM to avoid causing register spillage since we are processing more float values per CTA.
  - Use 4 local accumulators so the SM scheduler can interleave operations from different accumulators while waiting for others to load
    - Previously v5 had a single chain since each thread was processing 32 floats per thread.

Testing empirically, this configuration seems to reduce the errors on the large test case and leaderboard submission is passing.

Large test case error comparison:

| Submission | Size | Maximum error |
| --- | ---: | ---: |
| v5 | 52,428,800 | 4,352.0 |
| v6 | 52,428,800 | 2,048.0 |

v6 reduced the large-case error from v5 by 2,304, while keeping a similar benchmark runtime on the largest case.

Big caveat: after running the evaluation script repeatedly, there seems to be a big variance in the mean timings, usually +/- 3 us.

Benchmark timings:

| Size | H100 mean (ms) |
| ---: | ---: |
| 1,638,400 | 0.011 |
| 3,276,800 | 0.013 |
| 6,553,600 | 0.017 |
| 13,107,200 | 0.027 |
| 26,214,400 | 0.044 |
| 52,428,800 | 0.080 |


### References

- https://veitner.bearblog.dev/making-vector-sum-really-fast/
- https://github.com/pranjalssh/fast.cu/blob/main/sum.cu

