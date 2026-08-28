import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from src.schemas import FunctionDefinition
from src.engine import GenerationEngine


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Function Calling Inference Engine")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the output JSON will be saved."
    )
    return parser.parse_args()


def load_functions(filepath: Path) -> List[FunctionDefinition]:
    """Loads and validates function definitions."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [FunctionDefinition.model_validate(func) for func in data]
    except FileNotFoundError:
        print(f"Error: Function definitions file not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error validating function definitions: {e}")
        sys.exit(1)


def load_prompts(filepath: Path) -> List[Dict[str, Any]]:
    """Loads input prompts."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(item, dict) and 'prompt' in item for item in data):
            print(
                "Error: Input file must be a JSON array of objects with a 'prompt' key.")
            sys.exit(1)

        return data
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)


def main() -> None:
    """Main execution entry point."""
    args = parse_args()

    funcs_path = Path(args.functions_definition)
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading function definitions from {funcs_path}...")
    functions = load_functions(funcs_path)

    print(f"Loading prompts from {input_path}...")
    prompts = load_prompts(input_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        sys.exit(1)

    print(
        f"Successfully loaded {len(functions)} functions and {len(prompts)} prompts.")

    # 1. Initialize Engine
    print("Initializing Generation Engine (this will load the model into memory)...")
    try:
        engine = GenerationEngine(functions)
    except Exception as e:
        print(f"Failed to initialize Engine or LLM SDK: {e}")
        sys.exit(1)

# 2. Run the Generation Loop
    results = []
    print("\nStarting generation loop...")

    for i, p_data in enumerate(prompts):
        user_prompt = p_data.get("prompt", "")
        print(f"Processing [{i+1}/{len(prompts)}]: {user_prompt}")

        try:
            raw_call = engine.generate_function_call(user_prompt)

            formatted_result = {
                "prompt": user_prompt,
                "name": raw_call.get("name"),
                "parameters": raw_call.get("parameters", {})
            }
            results.append(formatted_result)
            print(
                f"  -> Extracted: {formatted_result['name']} with {len(formatted_result['parameters'])} params")

        except Exception as e:
            # TEMPORARY DEBUGGING FIX: Print the exact traceback
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 3. Save Final Results
    print(f"\nSaving {len(results)} valid results to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # We use indent=2 for clean, readable JSON
            json.dump(results, f, indent=2)
        print("Done! Output generated successfully.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
