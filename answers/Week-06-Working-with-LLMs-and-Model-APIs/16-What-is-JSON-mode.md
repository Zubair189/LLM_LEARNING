# What is JSON mode?

> Week 06 – Working with LLMs and Model APIs · Question 16

> Note: JSON mode is one **provider setting** that guarantees valid JSON, while [structured output](15-What-is-structured-output.md) is the **bigger goal** of getting data in your exact shape.

## In One Line
JSON mode is a setting that **forces the model to reply with valid JSON only**, with no extra chatty text.

## Simple Explanation
If you just ask a model "reply in JSON", it might say: "Sure! Here's the data: {...}". The extra words break `json.loads()`.

**JSON mode** fixes this. While the model writes, the system blocks any token that would make the text invalid JSON. So the reply can always be parsed.

What it **does** guarantee:
- The reply is **valid JSON** (correct brackets, quotes and commas).

What it does **not** guarantee:
- That the JSON has the **fields you want**. You asked for `name` and might get `full_name`.
- That the values have the **right types**.

So you still:
1. Describe the fields in your prompt.
2. **Validate** the result, for example with Pydantic.
3. Check the answer wasn't **cut off** by `max_tokens` (half a JSON object is invalid).

Where you find it:
- **OpenAI:** `response_format={"type": "json_object"}`.
- **Ollama:** `"format": "json"`.
- **Anthropic:** has no separate JSON-mode flag. Use a forced tool call with a schema, or its structured-outputs feature.

**Link to LLMs:** if you need exact fields, prefer schema-based structured output; JSON mode is the simpler, weaker option.

## Real-Life Example
JSON mode is like a form checker that only makes sure your handwriting is readable. It doesn't check that you wrote your name in the "name" box.

## Code Example
Why "valid JSON" is not enough.

```python
import json

def check(reply):
    try:
        return json.loads(reply)                     # valid JSON? -> Python dict
    except json.JSONDecodeError:
        return "cannot parse"

print(check('Sure! Here is the data: {"name": "Ana"}'))   # without JSON mode (chatty)
print(check('{"name": "Ana", "age": 31}'))                # with JSON mode
print(check('{"full_name": "Ana"}'))                      # valid JSON, but wrong keys!

# Output:
# cannot parse
# {'name': 'Ana', 'age': 31}
# {'full_name': 'Ana'}
```

**What happened:** the chatty reply couldn't be parsed. JSON mode prevents that, but the last reply shows valid JSON can still have the wrong field names. To turn it on in Ollama, add `"format": "json"` to the request body.

## Watch Out
- "JSON mode means my fields are correct" is wrong. It only means the text **parses**.
- Mention JSON and the fields in your prompt too. Some providers require it.

## Remember
- JSON mode = **always valid JSON**.
- It does **not** check fields or types.
- **Validate** the result, or use schema-based structured output.

## Related Questions
- [What is structured output?](15-What-is-structured-output.md)
- [What is JSON?](../Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
- [What is Ollama?](05-What-is-Ollama.md)
- [How do Python exceptions work?](../Week-05-Python-for-LLM-Engineering/08-How-do-Python-exceptions-work.md)
