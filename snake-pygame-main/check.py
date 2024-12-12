import numpy as np

def inspect_model(filepath="snake_model.npy"):
    """
    This file and method are used purely to make sure the data in our files is being saved correctly. Thus, this file/method is only for debugging purposes
    """
    try:
        # Load the model file
        model_data = np.load(filepath, allow_pickle=True)
        print(f"Model file contents: {model_data}")

        print(f"Shape of the loaded model data: {model_data.shape if hasattr(model_data, 'shape') else 'No shape attribute'}")

        if model_data.ndim == 1:
            print("Model file appears to be a 1D array or scalar, unable to iterate.")

        elif model_data.ndim == 2:
            for i, data in enumerate(model_data):
                print(f"\nLayer {i + 1} data:")
                print(data)

    except FileNotFoundError:
        print(f"No file found at {filepath}. Make sure the model has been saved.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

inspect_model("snake_model.npy")
