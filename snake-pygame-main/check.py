import numpy as np

def inspect_model(filepath="snake_model.npy"):
    try:
        # Load the model file
        model_data = np.load(filepath, allow_pickle=True)
        print(f"Model file contents: {model_data}")

        # Check the shape of the loaded model
        print(f"Shape of the loaded model data: {model_data.shape if hasattr(model_data, 'shape') else 'No shape attribute'}")

        if model_data.ndim == 1:
            print("Model file appears to be a 1D array or scalar, unable to iterate.")

        elif model_data.ndim == 2:
            # If it's a 2D array, iterate through it as expected
            for i, data in enumerate(model_data):
                print(f"\nLayer {i + 1} data:")
                print(data)

    except FileNotFoundError:
        print(f"No file found at {filepath}. Make sure the model has been saved.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

# Call this function to inspect the file
inspect_model("snake_model.npy")
