import json
from typing import List
from src.schemas import FunctionDefinition
import argparse
import sys
from pathlib import Path


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arguemnt Parser ")
    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/functions_definition.json",
                        help="Path to the input Prompts")

    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json",
                        help="Path to the input prompts JSON file.")
    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calling_results.json",
                        help="Path where the output JSON will be saved.")
    return parser.parse_args()


def load_functions(filepath: str) -> List[FunctionDefinition]:
    """
    Load and Validate function definations from json file
    """
    try:
        with open(filepath, "r") as file:
            data = json.load(file)

        return [FunctionDefinition.model_validate(func) for func in data]
    except FileNotFoundError:
        print(f"Error: Function definitions file not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error validating function definitions: {e}")
        sys.exit(1)


def load_prompts(filepath: Path) -> List[dict[str, any]]:
    """
    Loads input prompts from a JSON file.

    Args:
        filepath (Path): The path to the JSON file.

    Returns:
        List[Dict[str, Any]]: A list of
        dictionaries, each containing a 'prompt' key.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate that the file is a list of objects containing a "prompt" key
        if not isinstance(data, list) or not all(isinstance(item, dict)
                                                 and
                                                 'prompt' in item for
                                                 item in data):
            print(
                "Error: Input file must be a JSON"
                " array of objects with a 'prompt' key.")
            sys.exit(1)

        return data
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main Entry Point
    """

    args = args_parser()
    func_path = Path(args.functions_definition)
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading function definitions from {func_path}...")
    functions = load_functions(func_path)

    print(f"Loading prompts from {input_path}...")
    prompts = load_prompts(input_path)

    # Ensure the output directory exists so we don't crash when saving
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        sys.exit(1)

    print(
        f"Successfully loaded {len(functions)} "
        f"functions and {len(prompts)} prompts.")
    print("Ready to initialize LLM and start generation loop...")


if __name__ == "__main__":
    main()
