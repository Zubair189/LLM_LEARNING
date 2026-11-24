# What happens when a user sends a question to an LLM?

> Week 12 – Final Project and Revision · Question 3

## In One Line
Your app checks the question, builds a prompt (with history and documents), sends it to the model, which writes the answer **one token at a time**, and then your app checks, logs and returns it.

## Simple Explanation
Pressing "Send" starts a journey with three parts: your **backend** (your server) gets things ready, the **LLM provider** generates the answer, and your backend tidies up.

```
User → Backend checks → Cache? → Find documents → Build prompt → LLM writes token by token → Check answer → Log → User
```

**Step by step:**

**Part 1: your backend prepares**
1. **Check the user.** Are they logged in? Have they gone over their **rate limit** (a cap on requests)?
2. **Check the input.** Is it too long? Does it look like a prompt-injection trick?
3. **Look in the cache.** Was this exact question answered recently? If so, return the saved answer and stop.
4. **Find documents (RAG).** Turn the question into an embedding and search a vector database for the most relevant chunks.
5. **Build the prompt.** Combine the system prompt (rules), recent chat history, the retrieved chunks and the question. The model remembers nothing by itself, so everything it needs must be in this one prompt.

**Part 2: the LLM generates**

6. **Tokenize.** The provider splits the prompt into tokens. Input tokens are counted for cost and must fit the context window.
7. **Read the prompt.** The Transformer processes the whole prompt at once (this is called "prefill").
8. **Pick the next token.** The model gives a probability for every possible next token. Settings like **temperature** decide how adventurous the choice is.
9. **Repeat.** The new token is added to the end, and the model predicts the next one, again and again. It stops at an "end" token or when it hits `max_tokens`.
10. **Stream.** Tokens are sent back as they are made, so the user sees words appear quickly.

**Part 3: your backend finishes**

11. **Check the answer.** Is the format right? Does it contain private data? Is it supported by the documents?
12. **Log it.** Record speed, input/output tokens, cost and a request ID.
13. **Save it.** Store the turn in the chat history and the cache. Return the answer to the user.

**Where does the time go?** Usually the biggest part is step 9, writing the output, because it happens one token at a time. Long prompts also make step 7 slower.

## Real-Life Example
Ordering in a restaurant. The host checks your booking (login and limits). The waiter writes your order and adds notes from your past visits and today's specials (history and documents). The kitchen cooks one dish at a time (token by token). The waiter checks the plate before serving (output checks), and the manager records the order (logging).

## Watch Out
- The LLM doesn't remember your previous message unless your app **sends it again**.
- The LLM doesn't search the internet or your database by itself. It only sees what you put in the prompt or what tools you give it.

## Remember
- **Backend prepares → LLM generates → backend checks and logs.**
- The model is **stateless**; your app supplies memory and knowledge.
- Answers are generated **one token at a time**; streaming makes it feel fast.

## Related Questions
- [What is inference?](../Week-02-Introduction-to-LLMs-and-Generative-AI/11-What-is-inference.md)
- [What is the temperature parameter?](../Week-06-Working-with-LLMs-and-Model-APIs/12-What-is-the-temperature-parameter.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [How do you maintain conversation history?](../Week-11-Production-LLM-Applications/04-How-do-you-maintain-conversation-history.md)
- [What is the role of an API gateway?](../Week-11-Production-LLM-Applications/02-What-is-the-role-of-an-API-gateway.md)
- [Explain Transformer architecture.](02-Explain-Transformer-architecture.md)
