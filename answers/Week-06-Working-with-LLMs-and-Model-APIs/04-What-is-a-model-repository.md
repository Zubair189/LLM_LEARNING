# What is a model repository?

> Week 06 – Working with LLMs and Model APIs · Question 4

## In One Line
A model repository is a **storage place that keeps all of a model's files together**, with versions, so anyone can find and download it.

## Simple Explanation
A model is not just one file. To use it, you need several files that belong together:

1. **Weights:** the learned numbers, often in a `.safetensors` file. This is the biggest part.
2. **Config (`config.json`):** describes how the model is built, like how many layers it has.
3. **Tokenizer files:** the rules for cutting text into tokens (small pieces of text).
4. **Model card (`README.md`):** explains the licence, how to use it and its limits.

A model repository keeps these together. The word is used in two ways:
- **One repo** for one model, like `Qwen/Qwen2.5-0.5B-Instruct`.
- **A platform** that hosts many repos, like the Hugging Face Hub or the Ollama library.

Repos have **versions**, like saved checkpoints. You can **pin** (lock) a specific version so your app doesn't change when the author uploads new files.

**Link to LLMs:** when you run `ollama pull llama3.2` or load a model with `transformers`, you are downloading from a model repository.

## Real-Life Example
A model repository is like an app in an app store. It has a name, a version number, the files you install, and a description page. You install it by name and can choose a specific version.

## Code Example
```python
# pip install huggingface_hub
from huggingface_hub import list_repo_files, hf_hub_download
import json

repo = "Qwen/Qwen2.5-0.5B-Instruct"                      # repo name on the Hub
files = list_repo_files(repo)                            # list the files in it
print([f for f in files if f.endswith(".json")])         # show only the JSON files

path = hf_hub_download(repo_id=repo, filename="config.json")  # download one small file
cfg = json.load(open(path))                              # read it as a dict
print(cfg["model_type"], cfg["num_hidden_layers"])       # model family and layer count

# Output (e.g.):
# ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']
# qwen2 24
```

**What happened:** we looked inside a model repo and read its config file, without downloading the big weights file.

## Watch Out
- Weights without the matching tokenizer and config are useless. Keep them together.
- Prefer `.safetensors` files. Older `.bin` / `.pt` files can run hidden code when loaded, so only load them from people you trust.

## Remember
- A repo holds **weights + config + tokenizer + model card**.
- Repos have **versions**; pin one for stable apps.
- Hugging Face Hub and the Ollama library are **model repositories**.

## Related Questions
- [What is Hugging Face?](03-What-is-Hugging-Face.md)
- [What is Ollama?](05-What-is-Ollama.md)
- [What is an inference API?](07-What-is-an-inference-API.md)
- [What is JSON?](../Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
