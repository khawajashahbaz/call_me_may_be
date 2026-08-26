import numpy as np
from typing import Dict, List, Set, Optional
from pathlib import Path
import json
from enum import Enum, auto
from src.schemas import FunctionDefinition


class GrammarState(Enum):
    """Represents the current expected token type in the JSON generation."""
    EXPECT_START = auto()             # Expecting '{'
    EXPECT_NAME_KEY = auto()          # Expecting '"name"'
    EXPECT_NAME_COLON = auto()        # Expecting ':'
    EXPECT_NAME_VALUE = auto()        # Expecting '"fn_add_numbers"', etc.
    EXPECT_NAME_COMMA = auto()        # Expecting ','
    EXPECT_PARAMS_KEY = auto()        # Expecting '"parameters"'
    EXPECT_PARAMS_COLON = auto()      # Expecting ':'
    EXPECT_PARAMS_START = auto()      # Expecting '{'
    EXPECT_PARAM_KEY = auto()         # Expecting '"arg_name"' or '}'
    EXPECT_PARAM_COLON = auto()       # Expecting ':'
    EXPECT_PARAM_VALUE = auto()       # Expecting a number, string, etc.
    EXPECT_PARAM_COMMA_OR_END = auto()  # Expecting ',' or '}'
    EXPECT_END = auto()               # Expecting '}' (closing the main JSON)
    DONE = auto()                     # Generation complete


class JSONStateTracker:
    """
    Tracks the grammar state of the JSON object being generated token-by-token.
    """

    def __init__(self, functions: List[FunctionDefinition]):
        self.state = GrammarState.EXPECT_START
        self.functions = functions

        # State memory to know what we are currently parsing
        self.selected_function: Optional[FunctionDefinition] = None
        self.current_param_key: Optional[str] = None
        self.parsed_params: List[str] = []

        # We need to build a buffer of the raw string generated so far
        # to handle tokens that break a word in half (e.g., '"na' and 'me"')
        self.text_buffer = ""

    def get_allowed_strings(self) -> List[str]:
        """
        Returns a list of exactly what strings
        are legally allowed next
        based on the current grammar state.
        """
        if self.state == GrammarState.EXPECT_START:
            return ["{"]

        elif self.state == GrammarState.EXPECT_NAME_KEY:
            return ['"name"']

        elif self.state == GrammarState.EXPECT_NAME_COLON:
            return [":"]

        elif self.state == GrammarState.EXPECT_NAME_VALUE:
            # The LLM can ONLY output the
            # name of a function we actually defined
            return [f'"{f.name}"' for f in self.functions]

        elif self.state == GrammarState.EXPECT_NAME_COMMA:
            return [","]

        elif self.state == GrammarState.EXPECT_PARAMS_KEY:
            return ['"parameters"']

        elif self.state == GrammarState.EXPECT_PARAMS_COLON:
            return [":"]

        elif self.state == GrammarState.EXPECT_PARAMS_START:
            return ["{"]

        elif self.state == GrammarState.EXPECT_PARAM_KEY:
            # We can only accept keys defined in the selected function's schema
            if not self.selected_function:
                return ["}"]

            allowed_keys = [
                f'"{key}"' for key in self.selected_function.parameters.keys()
                if key not in self.parsed_params
            ]

            # If we've provided all parameters
            # (or it takes none), allow closing brace
            if not allowed_keys or len(self.parsed_params) == len(self.selected_function.parameters):
                allowed_keys.append("}")

            return allowed_keys

        # ... logic for EXPECT_PARAM_COLON, VALUE, etc. goes here ...

        return []

    def advance_state(self, new_text: str) -> None:
        """
        Updates the FSM state based on the newly completed token/string.
        """
        # This is where you will transition
        # from EXPECT_START -> EXPECT_NAME_KEY
        # once the "{" is fully formed.
        pass


class VocabManager:
    """
    Manages token-to-string mappings and filters vocabulary based on target prefixes.
    """

    def __init__(self, vocab_path: str):
        """
        Loads the vocabulary mapping from the SDK's JSON file.

        Args:
            vocab_path (str): Path to the vocabulary JSON file.
        """
        self.vocab_path = Path(vocab_path)
        self.id_to_token: Dict[int, str] = self._load_vocab()

    def _load_vocab(self) -> Dict[int, str]:
        """
        Parses the vocabulary JSON file.

        Returns:
            Dict[int, str]: Mapping from token ID to token string.
        """
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # Depending on SDK format, vocab can be {token_str: id} or {id_str: token_str}
        parsed: Dict[int, str] = {}
        for k, v in raw_vocab.items():
            if isinstance(v, int):
                # Format: {"token_string": token_id}
                parsed[v] = k
            else:
                # Format: {"token_id": "token_string"}
                parsed[int(k)] = str(v)
        return parsed

    def get_valid_token_ids(
        self, current_buffer: str, allowed_targets: List[str]
    ) -> Set[int]:
        """
        Finds all token IDs that produce a valid prefix for any allowed target.

        Args:
            current_buffer (str): The string accumulated so far in the current state.
            allowed_targets (List[str]): Full candidate strings permitted by the grammar.

        Returns:
            Set[int]: Set of allowed token IDs.
        """
        valid_ids: Set[int] = set()

        for token_id, token_str in self.id_to_token.items():
            candidate = current_buffer + token_str

            # Check if this candidate is a prefix of ANY target string
            # OR if an allowed target is a prefix of candidate (e.g. multi-token jump)
            for target in allowed_targets:
                if target.startswith(candidate) or candidate.startswith(target):
                    valid_ids.add(token_id)
                    break

        return valid_ids
