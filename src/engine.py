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
        """
        # 1. Encode the starting prompt
        formatted_prompt = f"{prompt}\nGenerate the JSON function call for the above prompt:"
        raw_input_ids = self.llm.encode(formatted_prompt)

        # --- 42 SDK FIX: Sanitize the output of encode() ---
        if hasattr(raw_input_ids, "tolist"):
            raw_input_ids = raw_input_ids.tolist()

        if isinstance(raw_input_ids, list) and len(raw_input_ids) > 0 and isinstance(raw_input_ids[0], list):
            input_ids = [int(tok) for tok in raw_input_ids[0]]
        else:
            input_ids = [int(tok) for tok in raw_input_ids]
        # ---------------------------------------------------

        # 2. Initialize the FSM for this specific generation (THIS WAS MISSING!)
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
                # If we expect a dynamic parameter value (like a number), we use a different check
                if fsm.state == GrammarState.EXPECT_PARAM_VALUE and fsm.selected_function and fsm.current_param_key:
                    param_type = fsm.selected_function.parameters[fsm.current_param_key].type
                    valid_token_ids = self.vocab_manager.get_value_token_ids(
                        current_buffer=fsm.text_buffer,
                        param_type=param_type
                    )

                if not valid_token_ids:
                    raise RuntimeError(
                        f"Grammar dead end: No tokens found for buffer '{fsm.text_buffer}' "
                        f"with allowed targets: {allowed_targets}"
                    )

            # 5. Get raw logits from the LLM
            raw_logits_tensor = self.llm.get_logits_from_input_ids(input_ids)

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
            next_token_id = int(sample_next_token(masked_logits))

            # 7. Append the token to the sequence
            input_ids.append(next_token_id)

            # 8. Decode the token to a string and update the buffer
            token_str = self.vocab_manager.id_to_token[next_token_id]
            fsm.text_buffer += token_str
            generated_json_string += token_str

            # 9. Check if the buffer has perfectly completed one of the target strings
            # Or if it's a dynamic value (number) that is now finished with a comma/brace
            if fsm.text_buffer in allowed_targets:
                fsm.advance_state(fsm.text_buffer)
            elif fsm.state == GrammarState.EXPECT_PARAM_VALUE:
                # If we are parsing a number, check if the LLM outputted the terminating comma or brace
                if fsm.text_buffer.endswith(",") or fsm.text_buffer.endswith("}"):
                    # Strip the terminating character, validate, and move state
                    value_str = fsm.text_buffer[:-1].strip()
                    fsm.advance_state(value_str)
                    # Immediately process the terminating character
                    fsm.text_buffer = fsm.text_buffer[-1]
                    fsm.advance_state(fsm.text_buffer)

        # 10. Final parsing
        try:
            return json.loads(generated_json_string)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse generated text into JSON. Raw text: '{generated_json_string}'"
            ) from e
