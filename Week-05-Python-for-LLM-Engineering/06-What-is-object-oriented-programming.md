# What is object-oriented programming?

> Week 05 – Python for LLM Engineering · Question 6

## Short Answer
Object-oriented programming (OOP) organises code around **objects**: bundles of data (attributes) and the behaviour that acts on it (methods), created from **classes**, which act as blueprints. Its four classic pillars are encapsulation, abstraction, inheritance and polymorphism. You use OOP constantly in LLM work, even if you never write a class yourself: `client = anthropic.Anthropic()` creates an object, and `client.messages.create(...)` calls a method on it.

## Detailed Explanation

| Pillar | Idea | LLM-engineering example |
|---|---|---|
| **Encapsulation** | Keep data and the methods that use it together; hide internals | A `ChatSession` that owns its `history` list |
| **Abstraction** | Expose a simple interface and hide complexity | `client.messages.create()` hides HTTP, auth and retries |
| **Inheritance** | A subclass reuses and extends a parent class | `OllamaProvider(LLMProvider)` |
| **Polymorphism** | Different classes, same method name, used interchangeably | Any provider with `.generate(prompt)` can be swapped in |

**Python specifics:**
- `__init__` initialises a new object. `self` is the instance.
- `@dataclass` generates `__init__`/`__repr__` for classes that mostly hold data.
- `@property` exposes computed attributes.
- "Duck typing": Python cares that an object *has* the right method, not what class it is. `typing.Protocol` lets you state that formally.
- **Prefer composition over deep inheritance.** A `RAGPipeline` *has* a retriever and an LLM rather than *being* one.

```
        LLMProvider (abstract: generate)
          /                     \
 ClaudeProvider            OllamaProvider
          \                     /
        ChatSession uses any provider (polymorphism)
```

## Example
**Analogy:** a class is a cookie cutter and objects are the cookies. Every cookie has the same shape (methods), but each can have its own toppings (attribute values).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

class EchoProvider(LLMProvider):          # fake provider for tests
    def generate(self, prompt: str) -> str:
        return f"Echo: {prompt}"

class ShoutProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return prompt.upper()

@dataclass
class ChatSession:
    provider: LLMProvider                 # composition: the session *has* a provider
    history: list[dict] = field(default_factory=list)

    def ask(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        reply = self.provider.generate(text)
        self.history.append({"role": "assistant", "content": reply})
        return reply

for provider in (EchoProvider(), ShoutProvider()):   # polymorphism
    session = ChatSession(provider)
    print(session.ask("hello"), "| turns:", len(session.history))
# Echo: hello | turns: 2
# HELLO | turns: 2
```

In a real app you would add a `ClaudeProvider` whose `generate` wraps `anthropic.Anthropic().messages.create(...)`. The `ChatSession` wouldn't change.

## Common Mistakes / Misconceptions
- **Forgetting `self`** in method definitions, or using a class attribute (shared by all instances) when you meant an instance attribute.
- **Deep inheritance trees.** Prefer small classes combined through composition.
- **Wrapping everything in classes.** A plain function is often simpler. Use a class when data and behaviour truly belong together, or when you keep state such as chat history.
- **`list = []` as a dataclass default** raises an error. Use `field(default_factory=list)`.

## Key Takeaways
- Classes are blueprints. Objects are instances holding their own state.
- Encapsulation, abstraction, inheritance and polymorphism are the four pillars.
- A common provider interface lets you swap cloud and local models and use fakes in tests.
- Favour `@dataclass` and composition for clean, testable code.

## Related Questions
- [What are functions?](05-What-are-functions.md)
- [What are modules and packages?](07-What-are-modules-and-packages.md)
- [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
