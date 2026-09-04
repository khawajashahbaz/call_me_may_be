"""
Finite State Machine and vocabulary management for constrained JSON decoding.
"""

from enum import Enum, auto
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.schemas import FunctionDefinition


class GrammarState(Enum):
    """Represents the current expected token type in JSON generation."""

    EXPECT_START = auto()
    EXPECT_NAME_KEY = auto()
    EXPECT_NAME_COLON = auto()
    EXPECT_NAME_VALUE = auto()
    EXPECT_NAME_COMMA = auto()
    EXPECT_PARAMS_KEY = auto()
    EXPECT_PARAMS_COLON = auto()
    EXPECT_PARAMS_START = auto()
    EXPECT_PARAM_KEY = auto()
    EXPECT_PARAM_COLON = auto()
    EXPECT_PARAM_VALUE = auto()
    EXPECT_PARAM_COMMA_OR_END = auto()
    EXPECT_END = auto()
    DONE = auto()


class JSONStateTracker:
    """Tracks and enforces JSON schema compliance token-by-token."""

    def __init__(self, functions: List[FunctionDefinition]):
        """
        Initializes the state tracker.

        Args:
            functions (List[FunctionDefinition]): Allowed function schemas.
        """
        self.state = GrammarState.EXPECT_START
        self.functions = functions
        self.selected_function: Optional[FunctionDefinition] = None
        self.current_param_key: Optional[str] = None
        self.parsed_params: List[str] = []
        self.text_buffer = ""

    def get_allowed_strings(self) -> List[str]:
        """
        Determines valid strings for the current grammar state.

        Returns:
            List[str]: Candidate strings permitted by the grammar.
        """
        if self.state == GrammarState.EXPECT_START:
            return ["{"]

        if self.state == GrammarState.EXPECT_NAME_KEY:
            return ['"name"']

        if self.state == GrammarState.EXPECT_NAME_COLON:
            return [":"]

        if self.state == GrammarState.EXPECT_NAME_VALUE:
            return [f'"{f.name}"' for f in self.functions]

        if self.state == GrammarState.EXPECT_NAME_COMMA:
            return [","]

        if self.state == GrammarState.EXPECT_PARAMS_KEY:
            return ['"parameters"']

        if self.state == GrammarState.EXPECT_PARAMS_COLON:
            return [":"]

        if self.state == GrammarState.EXPECT_PARAMS_START:
            return ["{"]

        if self.state == GrammarState.EXPECT_PARAM_KEY:
            if not self.selected_function:
                return ["}"]

            allowed_keys = [
                f'"{key}"'
                for key in self.selected_function.parameters.keys()
                if key not in self.parsed_params
            ]

            # If all parameters are filled, allow closing the parameter object
            if not allowed_keys or len(self.parsed_params) == len(
                self.selected_function.parameters
            ):
                return ["}"]

            return allowed_keys

        if self.state == GrammarState.EXPECT_PARAM_COLON:
            return [":"]

        if self.state == GrammarState.EXPECT_PARAM_COMMA_OR_END:
            if not self.selected_function:
                return ["}"]

            # If more parameters remain, expect a comma; otherwise, closing brace
            total_params = len(self.selected_function.parameters)
            if len(self.parsed_params) < total_params:
                return [","]
            return ["}"]

        if self.state == GrammarState.EXPECT_END:
            return ["}"]

        return []

    def advance_state(self, matched_text: str) -> None:
        """
        Advances the FSM to the next state upon matching an allowed string.

        Args:
            matched_text (str): The completed text segment that was matched.
        """
        self.text_buffer = ""

        if self.state == GrammarState.EXPECT_START:
            self.state = GrammarState.EXPECT_NAME_KEY

        elif self.state == GrammarState.EXPECT_NAME_KEY:
            self.state = GrammarState.EXPECT_NAME_COLON

        elif self.state == GrammarState.EXPECT_NAME_COLON:
            self.state = GrammarState.EXPECT_NAME_VALUE

        elif self.state == GrammarState.EXPECT_NAME_VALUE:
            func_name = matched_text.strip('"')
            self.selected_function = next(
                (f for f in self.functions if f.name == func_name), None
            )
            self.state = GrammarState.EXPECT_NAME_COMMA

        elif self.state == GrammarState.EXPECT_NAME_COMMA:
            self.state = GrammarState.EXPECT_PARAMS_KEY

        elif self.state == GrammarState.EXPECT_PARAMS_KEY:
            self.state = GrammarState.EXPECT_PARAMS_COLON

        elif self.state == GrammarState.EXPECT_PARAMS_COLON:
            self.state = GrammarState.EXPECT_PARAMS_START

        elif self.state == GrammarState.EXPECT_PARAMS_START:
            self.state = GrammarState.EXPECT_PARAM_KEY

        elif self.state == GrammarState.EXPECT_PARAM_KEY:
            if matched_text == "}":
                self.state = GrammarState.EXPECT_END
            else:
                self.current_param_key = matched_text.strip('"')
                self.parsed_params.append(self.current_param_key)
                self.state = GrammarState.EXPECT_PARAM_COLON

        elif self.state == GrammarState.EXPECT_PARAM_COLON:
            self.state = GrammarState.EXPECT_PARAM_VALUE

        elif self.state == GrammarState.EXPECT_PARAM_VALUE:
            self.state = GrammarState.EXPECT_PARAM_COMMA_OR_END

        elif self.state == GrammarState.EXPECT_PARAM_COMMA_OR_END:
            if matched_text == ",":
                self.state = GrammarState.EXPECT_PARAM_KEY
            else:
                self.state = GrammarState.EXPECT_END

        elif self.state == GrammarState.EXPECT_END:
            self.state = GrammarState.DONE

    def get_allowed_terminators(self) -> List[str]:
        """Determines if the value should end with a comma (more params) or brace (done)."""
        if not self.selected_function:
            return ["}"]
        if len(self.parsed_params) < len(self.selected_function.parameters):
            return [","]  # Still missing arguments
        return ["}"]      # All arguments filled


class VocabManager:
    """Manages token-to-string mappings and logit masking."""

    def __init__(self, vocab_path: str):
        """
        Loads vocabulary mapping from a JSON file.

        Args:
            vocab_path (str): Path to vocabulary JSON file.
        """
        self.vocab_path = Path(vocab_path)
        self.id_to_token: Dict[int, str] = self._load_vocab()

    def _load_vocab(self) -> Dict[int, str]:
        """
        Parses vocabulary mapping.

        Returns:
            Dict[int, str]: Mapping from token ID to token string.
        """
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            raw_vocab: Dict[str, Any] = json.load(f)

        parsed: Dict[int, str] = {}
        for k, v in raw_vocab.items():
            if isinstance(v, int):
                parsed[v] = k
            else:
                parsed[int(k)] = str(v)
        return parsed

    def get_valid_token_ids(
        self, current_buffer: str, allowed_targets: List[str]
    ) -> Set[int]:
        """Strictly filters tokens that build directly towards allowed targets."""
        valid_ids: Set[int] = set()

        for token_id, token_str in self.id_to_token.items():
            candidate = current_buffer + token_str
            for target in allowed_targets:
                # FIX: ONLY allow strict prefix matching. No overshooting!
                if target.startswith(candidate):
                    valid_ids.add(token_id)
                    break

        return valid_ids

    def get_value_token_ids(
        self, current_buffer: str, param_type: str, allowed_terminators: List[str]
    ) -> Set[int]:
        valid_ids: Set[int] = set()

        for token_id, token_str in self.id_to_token.items():
            candidate = current_buffer + token_str
            clean = candidate.replace('Ġ', '').replace('Ċ', ' ').strip()

            if param_type == "number":
                ends_with_term = False
                # 1. Always allow terminators if the number before it is valid
                for term in allowed_terminators:
                    if clean.endswith(term):
                        num_part = clean[:-1].strip()
                        if num_part in ["", "-"] or self._is_partial_number(num_part):
                            valid_ids.add(token_id)
                        ends_with_term = True
                        break

                # 2. SAFETY VALVE: Only allow adding more digits if the number is short
                is_too_long = len(clean) > 15
                if not ends_with_term and not is_too_long:
                    if clean in ["", "-"] or self._is_partial_number(clean):
                        valid_ids.add(token_id)

            elif param_type == "string":
                if not clean.startswith('"'):
                    if clean == "":
                        valid_ids.add(token_id)
                    continue

                parts = clean.replace('\\"', '').split('"')

                if len(parts) == 2:
                    valid_ids.add(token_id)
                elif len(parts) >= 3:
                    after_quote = parts[2].strip()
                    if after_quote == "":
                        valid_ids.add(token_id)
                    else:
                        for term in allowed_terminators:
                            if term.startswith(after_quote):
                                valid_ids.add(token_id)
                                break

        return valid_ids

    @staticmethod
    def _is_partial_number(text: str) -> bool:
        """Helper to verify if a string can form a valid number."""
        try:
            float(text)
            return True
        except ValueError:
            return False


def mask_logits(logits: np.ndarray, valid_token_ids: Set[int]) -> np.ndarray:
    """
    Sets logits of invalid tokens to negative infinity.

    Args:
        logits (np.ndarray): 1D array of model logits across vocabulary.
        valid_token_ids (Set[int]): Permitted token IDs.

    Returns:
        np.ndarray: Masked logits array.
    """
    masked = np.full_like(logits, -np.inf)
    if not valid_token_ids:
        return masked

    valid_indices = list(valid_token_ids)
    masked[valid_indices] = logits[valid_indices]
    return masked


def sample_next_token(masked_logits: np.ndarray) -> int:
    """
    Picks the highest-probability token using greedy selection.

    Args:
        masked_logits (np.ndarray): Masked logits array.

    Returns:
        int: The selected token ID.
    """
    return int(np.argmax(masked_logits))
