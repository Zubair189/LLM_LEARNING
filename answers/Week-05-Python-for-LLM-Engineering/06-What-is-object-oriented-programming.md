# What is object-oriented programming?

> Week 05 – Python for LLM Engineering · Question 6

## In One Line
Object-oriented programming (OOP) means organising code into **objects** that keep their **data and actions together**.

## Simple Explanation
A **class** is a blueprint. An **object** is one thing built from that blueprint.

Each object has:
- **Attributes:** its data, like `name` or `history`.
- **Methods:** functions that belong to it, like `ask()`.

Two Python words you will see:
- **`__init__`** runs automatically when a new object is created. It sets up the starting data.
- **`self`** means "this particular object".

OOP has four classic ideas:

1. **Encapsulation:** keep data and the code that uses it together.
2. **Abstraction:** show a simple button, hide the messy details.
3. **Inheritance:** a new class can reuse and extend an existing one.
4. **Polymorphism:** different classes can share the same method name, so you can swap them.

**Link to LLMs:** you use OOP all the time. `client = anthropic.Anthropic()` creates an object, and `client.messages.create(...)` calls one of its methods.

## Real-Life Example
A class is a cookie cutter and objects are the cookies. Every cookie has the same shape, but each can have different toppings. In the same way, every object has the same methods but its own data.

## Code Example
```python
class ChatSession:                                 # the blueprint
    def __init__(self, name):                      # runs when an object is created
        self.name = name                           # data (attribute)
        self.history = []                          # each object has its own list

    def ask(self, text):                           # behaviour (method)
        self.history.append(text)                  # remember the message
        return f"{self.name} got: {text}"          # fake reply

a = ChatSession("Bot A")                           # object 1
b = ChatSession("Bot B")                           # object 2, separate data
print(a.ask("Hello"))
print(a.ask("How are you?"))
print(b.ask("Hi"))
print(len(a.history), len(b.history))             # each keeps its own history

# Output:
# Bot A got: Hello
# Bot A got: How are you?
# Bot B got: Hi
# 2 1
```

**What happened:** two objects came from one class. Each kept its own chat history, so Bot A has 2 messages and Bot B has 1.

## Watch Out
- Methods need `self` as their first input. Forgetting it is a very common error.
- You don't need a class for everything. A plain function is often simpler.

## Remember
- **Class** = blueprint, **object** = thing built from it.
- Objects keep **data (attributes)** and **actions (methods)** together.
- LLM SDKs are built from objects, like `client.messages.create()`.

## Related Questions
- [What are functions?](05-What-are-functions.md)
- [What are modules and packages?](07-What-are-modules-and-packages.md)
- [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
