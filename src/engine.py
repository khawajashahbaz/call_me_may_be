import json
from typing import List, Dict, Any
import numpy as np

from llm_sdk import Small_LLM_Model
from src.schemas import FunctionDefinition
from src.decoder import (
    GrammarState,
    JSONStateTracker,
    VocabManager,
    mask_logits,
    sample_next_token
)


class GenerationEngine:
    """
    Manages the token-by-token generation process with constrained decoding.
    """

    def __init__(self, functions: List[FunctionDefinition]):
        """
        Initializes the LLM and the Vocabulary Manager.
        """
        self.functions = functions

        print("Loading LLM model...")
        self.llm = Small_LLM_Model()

        vocab_path = self.llm.get_path_to_vocab_file()
        self.vocab_manager = VocabManager(vocab_path)

    def generate_function_call(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Generates a valid JSON function call strictly matching the definitions.

        Args:
            prompt (str): The natural language input prompt.
            max_tokens (int): Safety limit to prevent infinite generation loops.

        Returns:
            Dict[str, Any]: The parsed JSON object of the function call.
        """
        # 1. Encode the starting prompt
        # We append a system instruction so the model knows what is expected.
        formatted_prompt = f"{prompt}\nGenerate the JSON function call for the above prompt:"
        input_ids = self.llm.encode(formatted_prompt)

        # 2. Initialize the FSM for this specific generation
        fsm = JSONStateTracker(self.functions)
        generated_json_string = ""

        for _ in range(max_tokens):
            if fsm.state == GrammarState.DONE:
                break

            # 3. Ask FSM: "What strings are legally allowed right now?"
            allowed_targets = fsm.get_allowed_strings()

            # 4. Ask VocabManager: "Which tokens can help me spell those strings?"
            valid_token_ids = self.vocab_manager.get_valid_token_ids(
                current_buffer=fsm.text_buffer,
                allowed_targets=allowed_targets
            )

            if not valid_token_ids:
                raise RuntimeError(
                    f"Grammar dead end: No tokens found for buffer '{fsm.text_buffer}' "
                    f"with allowed targets: {allowed_targets}"
                )

# 5. Get raw logits from the LLM
            # Wrap input_ids in an extra list to create a 2D batch: [[tokens]]
            raw_logits_tensor = self.llm.get_logits_from_input_ids([input_ids])

            # Safely extract the PyTorch tensor to a NumPy array
            if hasattr(raw_logits_tensor, "detach"):
                logits_np = raw_logits_tensor.detach().cpu().numpy()
            elif hasattr(raw_logits_tensor, "numpy"):
                logits_np = raw_logits_tensor.numpy()
            else:
                logits_np = np.array(raw_logits_tensor)

            # Squeeze out the batch dimension: [1, seq_len, vocab] -> [seq_len, vocab]
            logits_np = np.squeeze(logits_np)

            # We strictly want a 1D array of the VERY LAST token's predictions
            if logits_np.ndim > 1:
                next_token_logits = logits_np[-1, :]
            else:
                next_token_logits = logits_np

            # 6. Mask and sample the next token deterministically
            masked_logits = mask_logits(next_token_logits, valid_token_ids)

            # Ensure it is converted to a pure Python int, not a numpy/torch scalar
            next_token_id = int(sample_next_token(masked_logits))

            # 7. Append the pure integer token to the sequence
            input_ids.append(next_token_id)

            # 8. Decode the token to a string and update the buffer
            token_str = self.vocab_manager.id_to_token[next_token_id]
            fsm.text_buffer += token_str
            generated_json_string += token_str

            # 9. Check if the buffer has perfectly completed one of the target strings
            if fsm.text_buffer in allowed_targets:
                # The FSM will figure out what the next state should be and reset the buffer
                fsm.advance_state(fsm.text_buffer)

        # 10. Final parsing
        try:
            return json.loads(generated_json_string)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse generated text into JSON. Raw text: '{generated_json_string}'"
            ) from e
