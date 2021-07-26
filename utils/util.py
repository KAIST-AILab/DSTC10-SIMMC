import pickle

from metadata import FashionMetadata, FurnitureMetadata
from dialogue import AllDialogues
from scene import Scene

def load_pickle(pickle_type: str=None, pickle_specific_name: str=None):
    """
        pickle_type: should be one of
            [fashion_meta, furniture_meta, dial_train, dial_dev, dial_devtest, dial_test]
        pickle_specific_name:
            only if you want to load pickle other than pickle_type, specify pickle file name.
    """
    pickle_types = \
        {
            "fashion_meta": "fashion_meta.pkl",
            "furniture_meta": "furniture_meta.pkl",
            "dial_train": "dial_train.pkl",
            "dial_dev": "dial_dev.pkl",
            "dial_devtest": "dial_devtest.pkl",
            "dial_test": "dial_test.pkl"
        }

    pickle_name = pickle_types.get(pickle_type, None)
    if pickle_specific_name:
        pickle_name = pickle_specific_name
    if pickle_name is None:
        raise ValueError("Either pickle_type or pickle_specific_name argument should be given proper value")

    with open(pickle_name, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data






