# What is the difference between an open-weight model and a proprietary model?

> Week 02 – Introduction to LLMs and Generative AI · Question 13

## In One Line
An **open-weight** model lets you download its learned numbers (weights) and run it yourself, while a **proprietary** model keeps them secret and you can only use it through the company's app or API.

## Simple Explanation
Every model is defined by its **weights**: the billions of numbers it learned in training.

**Open-weight models** (for example Llama, Mistral, Qwen, Gemma, DeepSeek):
- You can **download** the weights and run the model on your own computer or server.
- You can **customise** it, for example by fine-tuning it on your own data.
- Your data can **stay on your machines**, which helps privacy.
- But **you** must run it: buy or rent the hardware and keep it working.

**Proprietary (closed) models** (for example Claude, GPT, Gemini):
- You use them only through an **API** or app. The company runs everything.
- They are often the **most capable** models and are easy to start with.
- But you depend on the company's prices, rules and model versions, and your data is sent to them (under their terms).

**"Open-weight" is not the same as "open-source".** Open-weight means only the weights are shared. Fully open-source would also share the training code and data details. Many "open" models also have licences with rules, so always read the licence.

**How to choose:**
- Want top quality and little setup? → **Proprietary**.
- Need strict privacy, offline use, or very high volume? → **Open-weight**.
- Many teams **mix both**.

Week 6 asks about **local vs. cloud**, which is about *where* a model runs. That is a separate question from *whether its weights are public*.

## Real-Life Example
A proprietary model is like a restaurant: great food, but you never see the recipe and the menu can change. An open-weight model is like being given the recipe book: you cook at home and change the dishes, but you buy the stove and do the washing up.

## Watch Out
- **Open-weight isn't free.** Hardware and engineering time cost money; at low volume an API is often cheaper.
- **Open-weight isn't "use for anything".** Licences can limit commercial use.

## Remember
- Open-weight: download it, run it, customise it.
- Proprietary: use it through an API; the company runs it.
- Open/closed (weights) and local/cloud (where it runs) are different questions.

## Related Questions
- [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is Ollama?](../Week-06-Working-with-LLMs-and-Model-APIs/05-What-is-Ollama.md)
- [What is Hugging Face?](../Week-06-Working-with-LLMs-and-Model-APIs/03-What-is-Hugging-Face.md)
- [What are model parameters?](./10-What-are-model-parameters.md)
