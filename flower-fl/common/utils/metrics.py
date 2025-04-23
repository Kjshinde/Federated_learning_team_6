# Custom evaluation metrics
import numpy as np

def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute accuracy given predicted and true label arrays.
    Returns a float between 0 and 1.
    """
    return (preds == labels).sum() / len(labels)
