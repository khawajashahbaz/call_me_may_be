*This project has been created as part of the 42 curriculum by mshahbaz.*

# Call Me Maybe

## Description

The goal of this project is to build a constrained decoding engine from scratch that bridges the gap between natural language and deterministic code. It forces a small Large Language Model (0.5B parameters) to output 100% perfectly formatted, parseable JSON for function calling.

Standard LLMs are probabilistic and frequently fail to output structured data reliably, often hallucinating invalid keys, missing closing braces, or appending conversational filler. This project intercepts the autoregressive generation loop token-by-token. By applying a custom Finite State Machine (FSM) to mask the model's vocabulary probabilities (logits), it mathematically guarantees the output adheres to a predefined JSON schema, all without relying on external constrained generation libraries.

## Instructions

This project uses `uv` as the primary package manager for dependency resolution and execution.

**Installation:**

1. Clone the repository:
```bash
git clone <your-repo-url>
cd call_me_maybe

```


2. Ensure you have `uv` and Python installed on your system.

**Execution:**
Run the main module directly from the root of the repository. It will load the definitions, process the tests, and write the output.

```bash
uv run python -m src --functions_definition data/input/functions_definition.json

```

## Algorithm explanation

The core of this project is an in-flight interception loop that controls the LLM generation process token-by-token. The algorithm follows these steps:

1. **Prompting & Encoding:** The user prompt and available function schemas are formatted and tokenized into numerical IDs using Byte-Pair Encoding (BPE).
2. **Forward Pass:** The model processes the input IDs and calculates raw probabilities (logits) for every token in its ~150,000 token vocabulary.
3. **FSM Query:** The `JSONStateTracker` checks the current syntactical state (e.g., waiting for a function name, waiting for a parameter value) and dictates what strings or characters are legally allowed next.
4. **Vocabulary Filtering:** The `VocabManager` performs prefix-matching, comparing the allowed strings against the entire vocabulary to identify valid Token IDs.
5. **Logit Masking:** The probabilities of all invalid Token IDs in the logits array are set to negative infinity.
6. **Sampling:** An `argmax` function selects the highest-probability token from the remaining valid options.
7. **Loop & Advance:** The selected token is appended to the sequence, the FSM state is updated based on the generated text, and the cycle repeats until the JSON object is properly closed.

## Design decisions

To ensure a clean, maintainable, and easily evaluable codebase, the architecture was modularized into distinct files:

* **Separation of Concerns:** `schemas.py` handles Pydantic validation, `fsm.py` handles pure string grammar rules without AI context, `vocab.py` acts as the translator between strings and tokens, and `engine.py` orchestrates the mathematical loop.
* **No External Constraints Libraries:** Libraries like Outlines or JSONformer were strictly avoided to fulfill project requirements and deeply understand the underlying mechanics.
* **Safety Valves:** Small LLMs can easily enter infinite hallucination loops (e.g., outputting zeroes endlessly). A hard length constraint (15 characters) was implemented inside the dynamic number generation logic to force the LLM to terminate runaway values.
* **Graceful Degradation:** The generation loop is wrapped in a `try/except` block inside `__main__.py` to ensure that if a specific prompt causes a fatal grammar dead-end, the script logs the error cleanly and continues to the next prompt instead of exiting the program.

## Performance analysis

* **Accuracy:** The deterministic logit mask guarantees 100% syntactical accuracy for the JSON structure. If the model outputs data, it is mathematically guaranteed to be parseable.
* **Reliability:** High, though limited by the baseline semantic reasoning of the 0.5B parameter model. If the model incorrectly selects a math function for a string-based prompt, the FSM forces it to generate a number, but the safety valves ensure the system safely closes the JSON rather than crashing.
* **Speed:** There is a slight computational overhead compared to unconstrained generation due to the string-matching required against the vocabulary on each step. However, this is optimized by isolating the FSM logic and only querying the full vocabulary when dynamic generation (like numbers or string values) is required.

## Challenges faced

* **BPE Subword Anomalies:** Language models do not generate clean words; they generate Byte-Pair subwords, often carrying space markers (like the 'G' character in Qwen's tokenizer).
* *Solution:* Implemented a sanitization step inside the `VocabManager` that strips BPE markers before evaluating the candidate string against the FSM's allowed targets.


* **String Boundaries and Trailing Commas:** The LLM would occasionally attempt to output a token that contained both the end of a string and a comma, even when the FSM only allowed a closing brace.
* *Solution:* The string validation logic was heavily tightened by splitting candidate tokens using unescaped quotes. This allowed the FSM to independently verify if the characters trailing a closed string built toward a legal terminator.


* **Context Blindness:** Initially, the FSM worked perfectly, but the LLM hallucinated wildly because it was not provided the function definitions in the prompt.
* *Solution:* Modified the engine to dynamically inject the available schemas into the system prompt before tokenization.



## Testing strategy

Validation was conducted iteratively using the provided `function_calling_tests.json`.
Before integrating the full loop, isolated demonstration scripts (such as `demo_basics.py`) were created to manually inspect tokenization behavior and view raw logit shapes. This allowed for precise debugging of the vocabulary prefix-matching logic. Finally, the system was stress-tested against prompts that required multi-parameter functions and prompts completely unrelated to the available tools to ensure the exception handling functioned correctly.

## Example usage

To execute the pipeline, run the module through `uv`:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json

```

**Expected Output Log:**

```text
Loading function definitions...
Successfully loaded 5 functions and 11 prompts.
Initializing Generation Engine...

Starting generation loop...
Processing [1/11]: What is the sum of 2 and 3?
  -> Extracted: fn_add_numbers with 2 params
Processing [2/11]: Greet shrek
  -> Extracted: fn_greet with 1 params
...

```

The parsed JSON calls are subsequently saved to `data/output/function_calling_results.json` in the following format:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]

```

## Resources

* **Andrej Karpathy - "Let's build the GPT Tokenizer":** Essential for understanding Byte-Pair Encoding and subword generation.
* **Hugging Face NLP Course (Text Generation):** Provided foundational knowledge on logits, autoregressive generation, and argmax sampling.
* **OpenAI Function Calling Documentation:** Used as a reference for structuring JSON schemas and understanding the intended mapping of natural language to arguments.
* **AI Usage:** Gemini was utilized as a thought partner and coding mentor throughout the development process. It was specifically used to debug FSM logic, fix BPE tokenization loopholes, architect the modular project structure, and generate Mermaid diagrams for architectural planning.