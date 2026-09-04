import json
from pathlib import Path
from typing import Any, Dict, List, Set


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
