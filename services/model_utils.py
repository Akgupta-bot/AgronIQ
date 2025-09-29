# services/model_utils.py
import numpy as np
import pickle
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from services import data_generators

# Optional: deep learning
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# -----------------------------
# RandomForest (baseline)
# -----------------------------
def train_model(seed=42):
    """
    Train a simple RandomForest on synthetic crop data (for Farmer dashboard).
    Returns (clf, crop_sample).
    """
    crop_sample = data_generators.generate_crop_image(seed=seed)
    X, y = crop_sample["X"], crop_sample["y"]

    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    clf.fit(X, y)
    return clf, crop_sample


def train_crop_model(X_train, y_train, seed=42, n_estimators=80, max_depth=12):
    """
    Train a RandomForest on provided dataset (for Researcher dashboard).
    """
    md = int(max_depth) if max_depth and max_depth > 0 else None
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=md,
        random_state=int(seed),
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X, y, seed=42, n_estimators=80, max_depth=12, test_size=0.2, use_cv=False):
    """
    Train a RandomForest model on input features.
    Returns dict: model, metrics, cv_scores.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

    md = int(max_depth) if max_depth and max_depth > 0 else None
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=md,
        random_state=int(seed),
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "classification_report": classification_report(y_test, y_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": clf.feature_importances_.tolist(),
    }

    cv_scores = None
    if use_cv:
        cv_scores = cross_val_score(clf, X, y, cv=5)

    return {"clf": clf, "metrics": metrics, "cv_scores": cv_scores}


# -----------------------------
# CNN for crop stress detection (patch-based)
# -----------------------------
def extract_patches(crop_sample, patch_size=16, stride=16):
    """
    Extract patches from multispectral crop image for CNN training.
    Each patch is (patch_size, patch_size, 6) with label = majority class.
    Returns: X (N, H, W, C), y (N,)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. CNN unavailable.")

    bands = np.stack([
        crop_sample["red"],
        crop_sample["green"],
        crop_sample["blue"],
        crop_sample["nir"],
        crop_sample["red_edge"],
        crop_sample["swir"],
    ], axis=-1)  # (H, W, 6)
    labels = crop_sample["labels"]  # (H, W)

    H, W, C = bands.shape
    X_patches, y_patches = [], []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = bands[i:i+patch_size, j:j+patch_size, :]
            patch_label = labels[i:i+patch_size, j:j+patch_size]
            maj = int(np.round(patch_label.mean()))  # majority vote
            X_patches.append(patch)
            y_patches.append(maj)

    X_patches = np.array(X_patches, dtype=np.float32)
    y_patches = np.array(y_patches, dtype=np.int32)
    return X_patches, y_patches


def build_cnn(input_shape=(16, 16, 6), num_classes=2):
    """
    Build a simple CNN for multispectral patch classification.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. CNN unavailable.")

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(crop_sample, patch_size=16, stride=16, epochs=5, batch_size=16):
    """
    Train CNN on patches extracted from synthetic crop image.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. CNN unavailable.")

    X, y = extract_patches(crop_sample, patch_size=patch_size, stride=stride)
    model = build_cnn(input_shape=X.shape[1:], num_classes=2)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


# -----------------------------
# LSTM for temporal anomaly detection
# -----------------------------
def build_lstm(input_shape=(30, 3)):
    """
    Build LSTM model for temporal anomaly detection in soil/weather data.
    Input shape: (timesteps, features)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. LSTM unavailable.")

    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(data, labels, epochs=10, batch_size=8):
    """
    Train LSTM on time-series data (soil/weather).
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. LSTM unavailable.")

    model = build_lstm(input_shape=(data.shape[1], data.shape[2]))
    model.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


# -----------------------------
# Stress map prediction
# -----------------------------
def predict_stress_map(clf, multispectral_sample):
    """
    Run per-pixel stress prediction on multispectral crop image.
    Uses features: RGB + NIR + RedEdge + SWIR + indices.
    Returns prediction mask (H,W).
    """
    H, W = multispectral_sample["red"].shape
    X_img = np.stack([
        multispectral_sample["red"].ravel(),
        multispectral_sample["green"].ravel(),
        multispectral_sample["blue"].ravel(),
        multispectral_sample["nir"].ravel(),
        multispectral_sample["red_edge"].ravel(),
        multispectral_sample["swir"].ravel(),
        multispectral_sample["ndvi"].ravel(),
        multispectral_sample["ndwi"].ravel(),
        multispectral_sample["savi"].ravel(),
        multispectral_sample["gci"].ravel(),
    ], axis=1)

    preds = clf.predict(X_img).reshape(H, W)
    return preds


# -----------------------------
# Continuous learning
# -----------------------------
def update_model(clf, X_new, y_new):
    """
    Update model with new data (continuous learning).
    Works for RandomForest with warm_start.
    """
    if hasattr(clf, "warm_start"):
        clf.warm_start = True
        clf.n_estimators += 10  # grow forest
        clf.fit(X_new, y_new)
    else:
        clf.fit(X_new, y_new)
    return clf


# -----------------------------
# Export
# -----------------------------
def export_model_pickle(clf):
    """
    Export trained model as pickle bytes for download.
    """
    buf = io.BytesIO()
    pickle.dump(clf, buf)
    buf.seek(0)
    return buf
