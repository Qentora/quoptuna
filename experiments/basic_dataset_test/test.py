from concurrent.futures import ProcessPoolExecutor

import numpy as np  # Import numpy for efficient array operations
from pmlb import classification_dataset_names, fetch_data


def fetch_binary_datasets(dataset_name):
    x, y = fetch_data(dataset_name, return_X_y=True)
    if len(np.unique(y)) == 2:  # Use numpy to check unique class labels efficiently
        return (dataset_name,len(x))
    return (None,None)


if __name__ == "__main__":
    binary_datasets = []

    with ProcessPoolExecutor(
            max_workers=20) as executor:  # Limit the number of workers
        results = list(
            executor.map(fetch_binary_datasets, classification_dataset_names))

    binary_datasets = [(dataset, length) for dataset, length in results
                       if dataset is not None]
    import pickle
    with open("experiments/basic_dataset_test/data/binary_datasets.pkl", "wb") as file:
        pickle.dump(binary_datasets, file)
    print("Binary classification datasets:", binary_datasets)
