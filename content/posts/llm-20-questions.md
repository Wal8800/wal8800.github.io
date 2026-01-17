---
title: "Running LLM locally"
date: 2024-10-09T17:00:48+13:00
draft: false
--- 

I was looking to run LLM locally to develop a bot to compete in [a 20 questions competition.](https://www.kaggle.com/competitions/llm-20-questions)

Found [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (a python binding on top of [llama-cpp](https://github.com/ggerganov/llama.cpp)) which enables us to run LLM with variety of hardwares. This is useful since I only have an old gaming GPU with little memory (8GB).

First, we need to download a LLM model. There are huge selection of LLM models that fine tuned and quantised in [hugging face model hub](https://huggingface.co/models) to run locally. The [LocalLLaMa subreddit](https://www.reddit.com/r/LocalLLaMA) is also a great resource about the latest LLM models, their performance benchmark and actual user experience.

I used quantised Llama 3.1 8B from [bartowski](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) since that was the latest model.
Using the `Q8_0` format as the GPU (Nvidia T4, 16GB VRAM) in Kaggle notebook could load the entire model (8.54 GB file size) and I can run it locally with my hardware (spilling some of the layers to CPU).

Loading the model is relatively straightforward.

```python
from llama_cpp import Llama

llm = Llama(
    model_path="llama-3.1-8b-q8-0-gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    n_ctx=4096,
    n_threads=11,
    n_gpu_layers=25,
    chat_format="llama-3",
)
```

- `n_ctx` is the number of tokens for context. Each model has it own limit and we can set it to `0` to use the default. However, the default limit for Llama 3.1 8B is quite big (128K tokens) and my local machine doesn't have enough memory for large context window. As a result, I set it to a smaller value `4096` which is sufficient for my prompt.
- `n_threads` is the number of CPU core the model will use, my machine has 12 cores so I set to `11` for 1 other core to do other tasks
- `n_gpu_layers` is the number of layer compute on the GPU. We can set it to `-1` to compute all layers at GPU. However since my GPU can't hold all the layers inside memory, I needed to set a specific amount. After some trial and error, I configured to 25 so the model will use most of the GPU to get a speed boost.

_Load the model with n_gpu_layer to - 1 and ran out of memory._

```console
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 2070, compute capability 7.5, VMM: yes
llm_load_tensors: ggml ctx size =    0.27 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 7605.34 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate backend buffer
llama_load_model_from_file: failed to load model
```

_Set to 1 to see how many layers we have._


```console
llm_load_tensors: offloading 1 repeating layers to GPU
llm_load_tensors: offloaded 1/33 layers to GPU
llm_load_tensors:        CPU buffer size =  8137.64 MiB
llm_load_tensors:      CUDA0 buffer size =   221.03 MiB
```

_Found the sweet spot with 25 layers._

```console
llm_load_tensors: offloading 25 repeating layers to GPU
llm_load_tensors: offloaded 25/33 layers to GPU
llm_load_tensors:        CPU buffer size =  8137.64 MiB
llm_load_tensors:      CUDA0 buffer size =  5525.79 MiB
```

Once the model is loaded, llama-cpp-python provides a [conversational interface](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#chat-completion) like OpenAI and Anthropic client SDK. We can construct a list of messages with different roles then pass the messages into the model.


```python
messages = [
    {
        "role": "system",
        "content": dedent("""
            Let's play 20 Questions to figure out a keyword. 
            The keyword is a specific place or person.
            Asks yes or no questions to guess the keyword.
        """),
    },
    {
        "role": "assistant",
        "content": "Is it a person?"
    },
    {
        "role": "user",
        "content": "yes"
    }
]
result = llm.create_chat_completion(messages)
extracted = result["choices"][0]["message"]["content"]
print("Bot reply: ", extracted)
```

```console
llama_perf_context_print:        load time =     870.59 ms
llama_perf_context_print: prompt eval time =       0.00 ms /    59 tokens (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /     6 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time =    1316.56 ms /    65 tokens
Bot reply:  Is it a historical figure?
```
