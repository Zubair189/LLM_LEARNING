# LLM Engineering – 90-Day Question Bank: Answers

Worked answers to every question in `LLM_Engineering_90_Day_Question_Bank.pdf`. Each question has its own Markdown file with a short answer, a detailed explanation, examples (an everyday analogy plus Python code where relevant), common mistakes, key takeaways and links to related questions.

See [QUESTION_BANK_CHANGES.md](QUESTION_BANK_CHANGES.md) for the questions added, duplicates found and suggested rewording.

**Contents:** 178 original questions + 13 bonus questions, across 12 weeks. Each week's folder has a `README.md` with a practical exercise and three "Explain Aloud" prompts.

## How to use

1. Open the week's `README.md` and try each question yourself **before** reading the answer.
2. Run the Python examples. Most need only the standard library or `numpy`; others need the libraries they import (e.g. `pandas`, `requests`, `chromadb`, `transformers`).
3. Hosted-API examples use the Anthropic Python SDK (`pip install anthropic`, set `ANTHROPIC_API_KEY`). Prices and model names change, so check the provider's docs.

> **About the example outputs:** examples in Weeks 1–4 and the pure-Python ones in Weeks 7–8 were run and their outputs are real. Examples that need API keys or heavy libraries (PyTorch, transformers, chromadb, faiss, sentence-transformers) were not run here; their outputs are marked approximate (`~`, `e.g.`).

## Weeks

### Week 1 – Introduction to Artificial Intelligence

[Week overview, exercise & revision prompts](Week-01-Introduction-to-Artificial-Intelligence/README.md)

1. [What is Artificial Intelligence (AI)?](Week-01-Introduction-to-Artificial-Intelligence/01-What-is-Artificial-Intelligence-AI.md)
2. [What is the difference between AI and Machine Learning?](Week-01-Introduction-to-Artificial-Intelligence/02-What-is-the-difference-between-AI-and-Machine-Learning.md)
3. [How does traditional programming differ from machine learning?](Week-01-Introduction-to-Artificial-Intelligence/03-How-does-traditional-programming-differ-from-machine-learning.md)
4. [What is supervised learning?](Week-01-Introduction-to-Artificial-Intelligence/04-What-is-supervised-learning.md)
5. [What is unsupervised learning?](Week-01-Introduction-to-Artificial-Intelligence/05-What-is-unsupervised-learning.md)
6. [What is reinforcement learning?](Week-01-Introduction-to-Artificial-Intelligence/06-What-is-reinforcement-learning.md)
7. [What is Deep Learning?](Week-01-Introduction-to-Artificial-Intelligence/07-What-is-Deep-Learning.md)
8. [What is a neural network?](Week-01-Introduction-to-Artificial-Intelligence/08-What-is-a-neural-network.md)
9. [What are neurons, weights and biases?](Week-01-Introduction-to-Artificial-Intelligence/09-What-are-neurons-weights-and-biases.md)
10. [What is a training dataset?](Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md)
11. [What is the difference between training and testing data?](Week-01-Introduction-to-Artificial-Intelligence/11-What-is-the-difference-between-training-and-testing-data.md)
12. [What is overfitting?](Week-01-Introduction-to-Artificial-Intelligence/12-What-is-overfitting.md)
13. [What is underfitting and the bias-variance tradeoff?](Week-01-Introduction-to-Artificial-Intelligence/13-BONUS-What-is-underfitting-and-the-bias-variance-tradeoff.md) *(bonus)*

### Week 2 – Introduction to LLMs and Generative AI

[Week overview, exercise & revision prompts](Week-02-Introduction-to-LLMs-and-Generative-AI/README.md)

1. [What is Generative AI?](Week-02-Introduction-to-LLMs-and-Generative-AI/01-What-is-Generative-AI.md)
2. [What is a Large Language Model (LLM)?](Week-02-Introduction-to-LLMs-and-Generative-AI/02-What-is-a-Large-Language-Model-LLM.md)
3. [How is an LLM different from a traditional chatbot?](Week-02-Introduction-to-LLMs-and-Generative-AI/03-How-is-an-LLM-different-from-a-traditional-chatbot.md)
4. [What does GPT stand for?](Week-02-Introduction-to-LLMs-and-Generative-AI/04-What-does-GPT-stand-for.md)
5. [What is a language model?](Week-02-Introduction-to-LLMs-and-Generative-AI/05-What-is-a-language-model.md)
6. [What is a token?](Week-02-Introduction-to-LLMs-and-Generative-AI/06-What-is-a-token.md)
7. [How does tokenization work?](Week-02-Introduction-to-LLMs-and-Generative-AI/07-How-does-tokenization-work.md)
8. [Why do LLMs use tokens instead of complete words?](Week-02-Introduction-to-LLMs-and-Generative-AI/08-Why-do-LLMs-use-tokens-instead-of-complete-words.md)
9. [What is a context window?](Week-02-Introduction-to-LLMs-and-Generative-AI/09-What-is-a-context-window.md)
10. [What are model parameters?](Week-02-Introduction-to-LLMs-and-Generative-AI/10-What-are-model-parameters.md)
11. [What is inference?](Week-02-Introduction-to-LLMs-and-Generative-AI/11-What-is-inference.md)
12. [Why do LLMs sometimes hallucinate?](Week-02-Introduction-to-LLMs-and-Generative-AI/12-Why-do-LLMs-sometimes-hallucinate.md)
13. [What is the difference between an open-weight model and a proprietary model?](Week-02-Introduction-to-LLMs-and-Generative-AI/13-What-is-the-difference-between-an-open-weight-model-and-a-proprietary-model.md)
14. [What is the difference between training and inference?](Week-02-Introduction-to-LLMs-and-Generative-AI/14-What-is-the-difference-between-training-and-inference.md)
15. [How are chat models trained (pretraining, instruction tuning, RLHF)?](Week-02-Introduction-to-LLMs-and-Generative-AI/15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md) *(bonus)*

### Week 3 – Mathematics and Neural Network Foundations

[Week overview, exercise & revision prompts](Week-03-Mathematics-and-Neural-Network-Foundations/README.md)

1. [What is a scalar?](Week-03-Mathematics-and-Neural-Network-Foundations/01-What-is-a-scalar.md)
2. [What is a vector?](Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
3. [What is a matrix?](Week-03-Mathematics-and-Neural-Network-Foundations/03-What-is-a-matrix.md)
4. [Why do neural networks use matrices?](Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
5. [What is a dot product?](Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
6. [What is probability?](Week-03-Mathematics-and-Neural-Network-Foundations/06-What-is-probability.md)
7. [What is a probability distribution?](Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
8. [What is an activation function?](Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
9. [What is ReLU?](Week-03-Mathematics-and-Neural-Network-Foundations/09-What-is-ReLU.md)
10. [What is softmax?](Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
11. [What is forward propagation?](Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
12. [What is a loss function?](Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
13. [What is gradient descent?](Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
14. [What is backpropagation?](Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
15. [Why do neural networks require optimization?](Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
16. [What is a learning rate?](Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md) *(bonus)*

### Week 4 – Transformer Architecture

[Week overview, exercise & revision prompts](Week-04-Transformer-Architecture/README.md)

1. [What problems do traditional RNNs have?](Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
2. [What is a Transformer?](Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
3. [Why did Transformers become important?](Week-04-Transformer-Architecture/03-Why-did-Transformers-become-important.md)
4. [What is attention?](Week-04-Transformer-Architecture/04-What-is-attention.md)
5. [What is self-attention?](Week-04-Transformer-Architecture/05-What-is-self-attention.md)
6. [What are Query, Key and Value?](Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
7. [What is an attention score?](Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
8. [What is multi-head attention?](Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
9. [What are positional encodings?](Week-04-Transformer-Architecture/09-What-are-positional-encodings.md)
10. [What are token embeddings?](Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
11. [What is an encoder?](Week-04-Transformer-Architecture/11-What-is-an-encoder.md)
12. [What is a decoder?](Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
13. [What is masked self-attention?](Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
14. [How does GPT use the Transformer architecture?](Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
15. [Why are Transformers effective at processing language?](Week-04-Transformer-Architecture/15-Why-are-Transformers-effective-at-processing-language.md)
16. [What is the KV cache and why does it speed up generation?](Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md) *(bonus)*

### Week 5 – Python for LLM Engineering

[Week overview, exercise & revision prompts](Week-05-Python-for-LLM-Engineering/README.md)

1. [Why is Python widely used in AI development?](Week-05-Python-for-LLM-Engineering/01-Why-is-Python-widely-used-in-AI-development.md)
2. [What are Python data types?](Week-05-Python-for-LLM-Engineering/02-What-are-Python-data-types.md)
3. [What is the difference between a list and a tuple?](Week-05-Python-for-LLM-Engineering/03-What-is-the-difference-between-a-list-and-a-tuple.md)
4. [What is a dictionary?](Week-05-Python-for-LLM-Engineering/04-What-is-a-dictionary.md)
5. [What are functions?](Week-05-Python-for-LLM-Engineering/05-What-are-functions.md)
6. [What is object-oriented programming?](Week-05-Python-for-LLM-Engineering/06-What-is-object-oriented-programming.md)
7. [What are modules and packages?](Week-05-Python-for-LLM-Engineering/07-What-are-modules-and-packages.md)
8. [How do Python exceptions work?](Week-05-Python-for-LLM-Engineering/08-How-do-Python-exceptions-work.md)
9. [What is JSON?](Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
10. [What is NumPy?](Week-05-Python-for-LLM-Engineering/10-What-is-NumPy.md)
11. [What is Pandas?](Week-05-Python-for-LLM-Engineering/11-What-is-Pandas.md)
12. [What is an HTTP request?](Week-05-Python-for-LLM-Engineering/12-What-is-an-HTTP-request.md)
13. [What are GET and POST requests?](Week-05-Python-for-LLM-Engineering/13-What-are-GET-and-POST-requests.md)
14. [How do you consume a REST API using Python?](Week-05-Python-for-LLM-Engineering/14-How-do-you-consume-a-REST-API-using-Python.md)
15. [How do you handle API errors?](Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
16. [What are async and await, and why do they matter for LLM apps?](Week-05-Python-for-LLM-Engineering/16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md) *(bonus)*
17. [What are virtual environments and how do you manage dependencies?](Week-05-Python-for-LLM-Engineering/17-BONUS-What-are-virtual-environments-and-how-do-you-manage-dependencies.md) *(bonus)*

### Week 6 – Working with LLMs and Model APIs

[Week overview, exercise & revision prompts](Week-06-Working-with-LLMs-and-Model-APIs/README.md)

1. [What is a pretrained model?](Week-06-Working-with-LLMs-and-Model-APIs/01-What-is-a-pretrained-model.md)
2. [What is a foundation model?](Week-06-Working-with-LLMs-and-Model-APIs/02-What-is-a-foundation-model.md)
3. [What is Hugging Face?](Week-06-Working-with-LLMs-and-Model-APIs/03-What-is-Hugging-Face.md)
4. [What is a model repository?](Week-06-Working-with-LLMs-and-Model-APIs/04-What-is-a-model-repository.md)
5. [What is Ollama?](Week-06-Working-with-LLMs-and-Model-APIs/05-What-is-Ollama.md)
6. [What is the difference between local and cloud LLMs?](Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
7. [What is an inference API?](Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
8. [What is an API key?](Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
9. [What is prompt engineering?](Week-06-Working-with-LLMs-and-Model-APIs/09-What-is-prompt-engineering.md)
10. [What is a system prompt?](Week-06-Working-with-LLMs-and-Model-APIs/10-What-is-a-system-prompt.md)
11. [What is a user prompt?](Week-06-Working-with-LLMs-and-Model-APIs/11-What-is-a-user-prompt.md)
12. [What is the temperature parameter?](Week-06-Working-with-LLMs-and-Model-APIs/12-What-is-the-temperature-parameter.md)
13. [What are top-k and top-p sampling?](Week-06-Working-with-LLMs-and-Model-APIs/13-What-are-top-k-and-top-p-sampling.md)
14. [What is response streaming?](Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
15. [What is structured output?](Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
16. [What is JSON mode?](Week-06-Working-with-LLMs-and-Model-APIs/16-What-is-JSON-mode.md)
17. [How do you manage API costs and rate limits?](Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
18. [What is few-shot prompting?](Week-06-Working-with-LLMs-and-Model-APIs/18-BONUS-What-is-few-shot-prompting.md) *(bonus)*

### Week 7 – Embeddings and Vector Databases

[Week overview, exercise & revision prompts](Week-07-Embeddings-and-Vector-Databases/README.md)

1. [What is an embedding?](Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
2. [Why do we convert text into vectors?](Week-07-Embeddings-and-Vector-Databases/02-Why-do-we-convert-text-into-vectors.md)
3. [What is a vector representation?](Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
4. [What is an embedding dimension?](Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
5. [What is the difference between word and sentence embeddings?](Week-07-Embeddings-and-Vector-Databases/05-What-is-the-difference-between-word-and-sentence-embeddings.md)
6. [What is semantic similarity?](Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
7. [What is cosine similarity?](Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
8. [What is Euclidean distance?](Week-07-Embeddings-and-Vector-Databases/08-What-is-Euclidean-distance.md)
9. [What is vector search?](Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
10. [What is a vector database?](Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
11. [How is a vector database different from a relational database?](Week-07-Embeddings-and-Vector-Databases/11-How-is-a-vector-database-different-from-a-relational-database.md)
12. [What is ChromaDB?](Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
13. [What is FAISS?](Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
14. [What is approximate nearest neighbor search?](Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
15. [How do you retrieve similar documents using embeddings?](Week-07-Embeddings-and-Vector-Databases/15-How-do-you-retrieve-similar-documents-using-embeddings.md)
16. [What is hybrid search (keyword + vector)?](Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md) *(bonus)*

### Week 8 – Retrieval Augmented Generation RAG

[Week overview, exercise & revision prompts](Week-08-Retrieval-Augmented-Generation-RAG/README.md)

1. [What is Retrieval-Augmented Generation (RAG)?](Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
2. [Why do LLMs need external knowledge?](Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
3. [How is RAG different from fine-tuning?](Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
4. [What is document ingestion?](Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
5. [What is document chunking?](Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
6. [Why do we divide documents into chunks?](Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
7. [What is chunk overlap?](Week-08-Retrieval-Augmented-Generation-RAG/07-What-is-chunk-overlap.md)
8. [What is a retrieval pipeline?](Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
9. [How does vector similarity retrieval work?](Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
10. [What is context injection?](Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
11. [What is a RAG pipeline?](Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
12. [What is hallucination in RAG?](Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
13. [What is metadata filtering?](Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
14. [What is the difference between retrieval and generation?](Week-08-Retrieval-Augmented-Generation-RAG/14-What-is-the-difference-between-retrieval-and-generation.md)
15. [How can we evaluate RAG answer quality?](Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
16. [What is reranking and why does it improve RAG?](Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md) *(bonus)*

### Week 9 – Fine tuning and Model Optimization

[Week overview, exercise & revision prompts](Week-09-Fine-tuning-and-Model-Optimization/README.md)

1. [What is model pretraining?](Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
2. [What is fine-tuning?](Week-09-Fine-tuning-and-Model-Optimization/02-What-is-fine-tuning.md)
3. [When should you fine-tune a model?](Week-09-Fine-tuning-and-Model-Optimization/03-When-should-you-fine-tune-a-model.md)
4. [What is supervised fine-tuning (SFT)?](Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md)
5. [What is a training dataset?](Week-09-Fine-tuning-and-Model-Optimization/05-What-is-a-training-dataset.md)
6. [What are training examples and labels?](Week-09-Fine-tuning-and-Model-Optimization/06-What-are-training-examples-and-labels.md)
7. [What is LoRA?](Week-09-Fine-tuning-and-Model-Optimization/07-What-is-LoRA.md)
8. [What is QLoRA?](Week-09-Fine-tuning-and-Model-Optimization/08-What-is-QLoRA.md)
9. [What is quantization?](Week-09-Fine-tuning-and-Model-Optimization/09-What-is-quantization.md)
10. [What is 4-bit quantization?](Week-09-Fine-tuning-and-Model-Optimization/10-What-is-4-bit-quantization.md)
11. [What is full fine-tuning?](Week-09-Fine-tuning-and-Model-Optimization/11-What-is-full-fine-tuning.md)
12. [How does LoRA reduce memory requirements?](Week-09-Fine-tuning-and-Model-Optimization/12-How-does-LoRA-reduce-memory-requirements.md)
13. [What is catastrophic forgetting?](Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md)
14. [How do you evaluate a fine-tuned model?](Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
15. [Can RAG replace fine-tuning?](Week-09-Fine-tuning-and-Model-Optimization/15-Can-RAG-replace-fine-tuning.md)
16. [What is preference tuning (RLHF and DPO)?](Week-09-Fine-tuning-and-Model-Optimization/16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md) *(bonus)*

### Week 10 – AI Agents and Tool Calling

[Week overview, exercise & revision prompts](Week-10-AI-Agents-and-Tool-Calling/README.md)

1. [What is an AI agent?](Week-10-AI-Agents-and-Tool-Calling/01-What-is-an-AI-agent.md)
2. [What is the difference between an LLM and an AI agent?](Week-10-AI-Agents-and-Tool-Calling/02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
3. [What is tool calling?](Week-10-AI-Agents-and-Tool-Calling/03-What-is-tool-calling.md)
4. [What is function calling?](Week-10-AI-Agents-and-Tool-Calling/04-What-is-function-calling.md)
5. [How does an LLM decide which tool to use?](Week-10-AI-Agents-and-Tool-Calling/05-How-does-an-LLM-decide-which-tool-to-use.md)
6. [What is agent memory?](Week-10-AI-Agents-and-Tool-Calling/06-What-is-agent-memory.md)
7. [What is short-term memory?](Week-10-AI-Agents-and-Tool-Calling/07-What-is-short-term-memory.md)
8. [What is long-term memory?](Week-10-AI-Agents-and-Tool-Calling/08-What-is-long-term-memory.md)
9. [What is an agent workflow?](Week-10-AI-Agents-and-Tool-Calling/09-What-is-an-agent-workflow.md)
10. [What is a multi-agent system?](Week-10-AI-Agents-and-Tool-Calling/10-What-is-a-multi-agent-system.md)
11. [What is an agent execution loop?](Week-10-AI-Agents-and-Tool-Calling/11-What-is-an-agent-execution-loop.md)
12. [What happens when a tool call fails?](Week-10-AI-Agents-and-Tool-Calling/12-What-happens-when-a-tool-call-fails.md)
13. [What are human-in-the-loop systems?](Week-10-AI-Agents-and-Tool-Calling/13-What-are-human-in-the-loop-systems.md)
14. [How do you prevent an agent from performing unauthorized operations?](Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
15. [What are the limitations of autonomous agents?](Week-10-AI-Agents-and-Tool-Calling/15-What-are-the-limitations-of-autonomous-agents.md)
16. [What is the Model Context Protocol (MCP)?](Week-10-AI-Agents-and-Tool-Calling/16-BONUS-What-is-the-Model-Context-Protocol-MCP.md) *(bonus)*

### Week 11 – Production LLM Applications

[Week overview, exercise & revision prompts](Week-11-Production-LLM-Applications/README.md)

1. [How do you design an LLM-powered application?](Week-11-Production-LLM-Applications/01-How-do-you-design-an-LLM-powered-application.md)
2. [What is the role of an API gateway?](Week-11-Production-LLM-Applications/02-What-is-the-role-of-an-API-gateway.md)
3. [How does FastAPI help in AI applications?](Week-11-Production-LLM-Applications/03-How-does-FastAPI-help-in-AI-applications.md)
4. [How do you maintain conversation history?](Week-11-Production-LLM-Applications/04-How-do-you-maintain-conversation-history.md)
5. [What is context management?](Week-11-Production-LLM-Applications/05-What-is-context-management.md)
6. [What is prompt injection?](Week-11-Production-LLM-Applications/06-What-is-prompt-injection.md)
7. [What is an indirect prompt injection attack?](Week-11-Production-LLM-Applications/07-What-is-an-indirect-prompt-injection-attack.md)
8. [How do you protect API keys?](Week-11-Production-LLM-Applications/08-How-do-you-protect-API-keys.md)
9. [What is rate limiting?](Week-11-Production-LLM-Applications/09-What-is-rate-limiting.md)
10. [What is response caching?](Week-11-Production-LLM-Applications/10-What-is-response-caching.md)
11. [What is token consumption?](Week-11-Production-LLM-Applications/11-What-is-token-consumption.md)
12. [How do you reduce LLM API costs?](Week-11-Production-LLM-Applications/12-How-do-you-reduce-LLM-API-costs.md)
13. [What is LLM evaluation?](Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)
14. [What is hallucination detection?](Week-11-Production-LLM-Applications/14-What-is-hallucination-detection.md)
15. [What is observability?](Week-11-Production-LLM-Applications/15-What-is-observability.md)
16. [How do you monitor latency and failures?](Week-11-Production-LLM-Applications/16-How-do-you-monitor-latency-and-failures.md)
17. [Why should AI-generated SQL be restricted?](Week-11-Production-LLM-Applications/17-Why-should-AI-generated-SQL-be-restricted.md)
18. [How do you implement human approval for sensitive operations?](Week-11-Production-LLM-Applications/18-How-do-you-implement-human-approval-for-sensitive-operations.md)
19. [What are guardrails in LLM applications?](Week-11-Production-LLM-Applications/19-BONUS-What-are-guardrails-in-LLM-applications.md) *(bonus)*

### Week 12 – Final Project and Revision

[Week overview, exercise & revision prompts](Week-12-Final-Project-and-Revision/README.md)

1. [Explain the complete lifecycle of an LLM.](Week-12-Final-Project-and-Revision/01-Explain-the-complete-lifecycle-of-an-LLM.md)
2. [Explain Transformer architecture.](Week-12-Final-Project-and-Revision/02-Explain-Transformer-architecture.md)
3. [What happens when a user sends a question to an LLM?](Week-12-Final-Project-and-Revision/03-What-happens-when-a-user-sends-a-question-to-an-LLM.md)
4. [Explain embeddings and vector search.](Week-12-Final-Project-and-Revision/04-Explain-embeddings-and-vector-search.md)
5. [Explain the complete RAG pipeline.](Week-12-Final-Project-and-Revision/05-Explain-the-complete-RAG-pipeline.md)
6. [Compare RAG and fine-tuning.](Week-12-Final-Project-and-Revision/06-Compare-RAG-and-fine-tuning.md)
7. [Explain LoRA and QLoRA.](Week-12-Final-Project-and-Revision/07-Explain-LoRA-and-QLoRA.md)
8. [Explain AI agent architecture.](Week-12-Final-Project-and-Revision/08-Explain-AI-agent-architecture.md)
9. [How does function calling work?](Week-12-Final-Project-and-Revision/09-How-does-function-calling-work.md)
10. [How do you secure an LLM application?](Week-12-Final-Project-and-Revision/10-How-do-you-secure-an-LLM-application.md)
11. [How do you evaluate response quality?](Week-12-Final-Project-and-Revision/11-How-do-you-evaluate-response-quality.md)
12. [How do you deploy an LLM application in production?](Week-12-Final-Project-and-Revision/12-How-do-you-deploy-an-LLM-application-in-production.md)
13. [Capstone: design an end-to-end document Q&A assistant](Week-12-Final-Project-and-Revision/13-BONUS-Capstone-design-an-end-to-end-document-QA-assistant.md) *(bonus)*
