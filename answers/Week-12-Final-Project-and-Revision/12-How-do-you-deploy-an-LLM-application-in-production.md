# How do you deploy an LLM application in production?

> Week 12 – Final Project and Revision · Question 12

## In One Line
You package your app as a small web service in a container, run several copies behind a gateway, keep data and secrets in proper stores, and add reliability, cost control, monitoring and safe releases.

## Simple Explanation
"Deploying" means putting your app online so real users can use it reliably. An LLM app has a few extra needs: model calls are slow, cost money per token, and can fail.

```
Users → Gateway (login, limits) → Load balancer → App copies (FastAPI in Docker) → LLM provider
                                                       ├─ Redis (cache, limits, sessions)
                                                       ├─ Vector DB (documents)  ├─ Database (users, history)
                                                       └─ Logs, metrics, traces → dashboards & alerts
Release path: Tests → Evaluation → Staging → Small rollout (5%) → Everyone
```

**Step by step:**

1. **Package it.** Write the app with **FastAPI**, put it in a **Docker container** (a box with your code and everything it needs, so it runs the same everywhere), and pin your library versions.
2. **Keep the app stateless.** Don't keep chat history or rate-limit counts inside the app's memory. Store them in **Redis** or a database. Then you can run many copies of the app, and any copy can handle any request.
3. **Put a front door in place.** An **API gateway** handles login, HTTPS and rate limits. A **load balancer** spreads traffic across the copies. Add more copies when traffic grows (**autoscaling**).
4. **Handle secrets.** API keys come from a **secrets manager** as environment variables, never from code.
5. **Make it reliable.**
   - **Timeouts** so a stuck call doesn't hang forever.
   - **Retries** with growing waits for temporary errors (like 429 or 5xx).
   - A **backup model** or cached answer if the main model is down.
   - **Streaming**, so users see words quickly.
   - A **health check** URL (like `/healthz`) so the platform knows each copy is alive.
6. **Control cost.** Cache answers, use cheaper models for easy tasks, set `max_tokens`, and give each user a budget.
7. **Watch it.** Log every request with an ID, tokens, cost and speed. Track p95 speed, error rate, cache hit rate, daily cost and quality signals like thumbs-down. Set alerts.
8. **Release safely.** Treat prompts like code: keep versions. In your CI/CD pipeline (automatic build-and-release steps), run your **evaluation test set**, and block the release if quality drops. Then go to staging, then a small share of users (**canary**), then everyone. Be ready to roll back.

**Three hosting choices:**
- **Hosted API model** (simplest): you only deploy your app; the provider runs the model.
- **Self-hosted model:** you run open-weight models on your own GPUs (e.g. with vLLM). You get more control, but you also own all the scaling and upkeep.
- **Mix:** a small local model for easy tasks and a hosted model for hard ones.

## Real-Life Example
Opening a restaurant chain. The recipe (prompt + model) is only part of it. You also need identical kitchens (containers), a host managing the queue (gateway and rate limits), a pantry (databases and cache), health inspections before opening (evaluation), a soft opening (canary release), and a manager watching wait times and complaints (monitoring).

## Watch Out
- Keeping state inside the app's memory breaks as soon as you run more than one copy.
- "The server is up" doesn't mean things are fine. An LLM app can be running while giving wrong or very expensive answers, so watch quality and cost too.

## Remember
- **Stateless container + gateway + proper data stores.**
- Reliability: **timeouts, retries, backup model, streaming**.
- Release with an **evaluation gate** and a **gradual rollout**, then keep monitoring.

## Related Questions
- [How do you design an LLM-powered application?](../Week-11-Production-LLM-Applications/01-How-do-you-design-an-LLM-powered-application.md)
- [How does FastAPI help in AI applications?](../Week-11-Production-LLM-Applications/03-How-does-FastAPI-help-in-AI-applications.md)
- [How do you reduce LLM API costs?](../Week-11-Production-LLM-Applications/12-How-do-you-reduce-LLM-API-costs.md)
- [How do you monitor latency and failures?](../Week-11-Production-LLM-Applications/16-How-do-you-monitor-latency-and-failures.md)
- [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [How do you secure an LLM application?](10-How-do-you-secure-an-LLM-application.md)
