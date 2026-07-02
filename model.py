"""
model.py
--------
Model loading and inference helpers for the pneumonia detection app.

Why this file is separate from app.py:
    Any preprocessing logic used at inference time MUST match exactly what
    was done at training time — otherwise predictions will be silently wrong.
    Keeping it in a dedicated module means we can see and verify the pipeline
    without wading through Streamlit UI code.

Training-time pipeline (from the notebook):
    - Read image with keras.utils.image_dataset_from_directory
    - Resize to (224, 224) with 3 channels (RGB)
    - Apply tf.keras.applications.efficientnet.preprocess_input
      (this internally handles the EfficientNet-specific scaling)
    - Feed into the fine-tuned EfficientNetB0 model
    - Output: single sigmoid probability (0 to 1)
        - Close to 0 -> Normal
        - Close to 1 -> Pneumonia

This module replicates that pipeline exactly for single-image inference.
"""

import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------------------------------------------
# Constants matching the training-time configuration
# -------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_model.keras"
IMG_SIZE = (224, 224)  # EfficientNetB0 input size
CLASS_NAMES = ["Normal", "Pneumonia"]  # 0=Normal, 1=Pneumonia (alphabetical)


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
def load_pneumonia_model():
    """
    Load the trained EfficientNetB0 model from disk.

    Returns
    -------
    tf.keras.Model
        The compiled model, ready for inference.
    """
    return tf.keras.models.load_model(str(MODEL_PATH))


# -------------------------------------------------------------------
# Image preprocessing
# -------------------------------------------------------------------
def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Convert a Streamlit-uploaded image into the (1, 224, 224, 3) tensor
    the model expects.

    Steps:
        1. Read the raw bytes from the upload
        2. Open as PIL image, convert to RGB (X-rays are grayscale but the
           model was trained on 3-channel RGB — replicating the grayscale
           across all 3 channels is what the training pipeline effectively did)
        3. Resize to 224x224 (bilinear by default in PIL, matching Keras)
        4. Convert to a numpy array of shape (224, 224, 3), dtype float32
        5. Add a batch dimension -> (1, 224, 224, 3)
        6. Apply EfficientNet's preprocess_input (internal normalisation)

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile or file-like
        The X-ray image the user has uploaded.

    Returns
    -------
    np.ndarray
        Preprocessed batch tensor, ready for model.predict()
    """
    # Step 1-2: Read + open + force RGB
    if hasattr(uploaded_file, "read"):
        image_bytes = uploaded_file.read()
    else:
        image_bytes = uploaded_file
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step 3: Resize
    pil_image = pil_image.resize(IMG_SIZE)

    # Step 4-5: Numpy + batch dimension
    arr = np.array(pil_image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    # Step 6: EfficientNet-specific preprocessing (identical to training)
    arr = preprocess_input(arr)

    return arr


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def predict(model, image_tensor: np.ndarray) -> float:
    """
    Run the model on a preprocessed image tensor and return the raw
    pneumonia probability.

    Parameters
    ----------
    model : tf.keras.Model
    image_tensor : np.ndarray
        Shape (1, 224, 224, 3), already preprocessed.

    Returns
    -------
    float
        Probability the X-ray shows pneumonia, in [0, 1].
    """
    prob = model.predict(image_tensor, verbose=0)
    return float(prob.flatten()[0])


def classify(probability: float, threshold: float) -> dict:
    """
    Turn the raw probability into a verdict + confidence, given a
    user-selected decision threshold.

    Parameters
    ----------
    probability : float
        Raw sigmoid output in [0, 1].
    threshold : float
        Decision cutoff. If probability > threshold, we predict pneumonia.

    Returns
    -------
    dict with keys:
        - verdict : "Pneumonia" or "Normal"
        - confidence : model's confidence in the CHOSEN class (not always the raw prob)
        - probability : the raw pneumonia probability
        - threshold : the threshold used
    """
    is_pneumonia = probability > threshold
    verdict = "Pneumonia" if is_pneumonia else "Normal"
    # If we predicted pneumonia, confidence is the raw probability.
    # If we predicted normal, confidence is 1 - probability.
    confidence = probability if is_pneumonia else (1.0 - probability)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "probability": probability,
        "threshold": threshold,
    }
