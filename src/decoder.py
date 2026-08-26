from enum import Enum, auto
from typing import List, Optional
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
        # This is where you will transition from EXPECT_START -> EXPECT_NAME_KEY
        # once the "{" is fully formed.
        pass
