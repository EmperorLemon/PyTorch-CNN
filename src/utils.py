import os

def prompt_user(model_path: str) -> int:
    # Ask user if they want to train a new model or load an existing one
    user_choice = input("\nDo you want to train a new model or load an existing one? (train/load): ")

    if user_choice.lower() == "train":
        return 0
    elif user_choice.lower() == "load":
        if os.path.exists(model_path):
            return 1
        else:
            print("No saved model found. Training a new model.")
            return 0
    else:
        print("Invalid choice. Training a new model.")
        return 0