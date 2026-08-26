import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Train an analytics model with custom parameters.")

    # Positional (Required)
    parser.add_argument("data_path", help="Path to the training dataset")

    # Optional with default and type constraints
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")

    # Optional with restricted choices
    parser.add_argument(
        "--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimization algorithm")

    # Boolean flag (True if --verbose is typed, False otherwise)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable detailed logging during training")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments using args.name
    print(f"Loading dataset: {args.data_path}")
    print(
        f"Configuration -> Epochs: {args.epochs} | Optimizer: {args.optimizer}")

    if args.verbose:
        print("[DEBUG] Verbose mode is activated. Tracking detailed metrics...")


if __name__ == "__main__":
    main()
