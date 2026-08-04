"""
Export Keras model weights to NumPy .npz files.
Run this script LOCALLY where full TensorFlow is installed.
The resulting .npz files should be committed to the repo for deployment.

Architecture (all 4 models):
  Dense(64, relu) -> Dropout -> Dense(32, relu) -> Dense(1, linear)

Each .npz file contains:
  w0, b0  (Dense 64 weights & biases)
  w1, b1  (Dense 32 weights & biases)
  w2, b2  (Dense 1 weights & biases)
"""

import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "nhl_point_predictor": os.path.join(BASE_DIR, "nhl_point_predictor.keras"),
    "nhl_point_predictorv2": os.path.join(BASE_DIR, "nhl_point_predictorv2.keras"),
    "nhl_points_predictorv3": os.path.join(BASE_DIR, "nhl_points_predictorv3.keras"),
    "nhl_points_predictorv4": os.path.join(BASE_DIR, "nhl_points_predictorv4.keras"),
}


def export_weights(name, keras_path):
    npz_path = os.path.join(BASE_DIR, f"{name}.npz")
    print(f"Exporting {os.path.basename(keras_path)} ...")

    model = tf.keras.models.load_model(keras_path)

    # Extract weights from Dense layers only (skip Dropout — it has no weights
    # and is a no-op at inference time)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    weight_dict = {}
    for i, layer in enumerate(dense_layers):
        weights, biases = layer.get_weights()
        weight_dict[f"w{i}"] = weights
        weight_dict[f"b{i}"] = biases
        print(f"  Layer {i}: {layer.name} — weights {weights.shape}, biases {biases.shape}")

    np.savez(npz_path, **weight_dict)

    size_kb = os.path.getsize(npz_path) / 1024
    print(f"  -> Saved {os.path.basename(npz_path)} ({size_kb:.1f} KB)")

    # Verify: run a sample prediction through both paths
    sample_input = np.random.randn(1, model.input_shape[1]).astype(np.float32)
    keras_pred = model.predict(sample_input, verbose=0)[0][0]

    # Pure NumPy forward pass
    x = sample_input
    for i in range(len(dense_layers)):
        x = x @ weight_dict[f"w{i}"] + weight_dict[f"b{i}"]
        if i < len(dense_layers) - 1:  # ReLU on all but last layer
            x = np.maximum(x, 0)
    numpy_pred = float(x[0][0])

    match = np.isclose(keras_pred, numpy_pred, atol=1e-5)
    print(f"  Verification: Keras={keras_pred:.6f}, NumPy={numpy_pred:.6f} — {'PASS' if match else 'MISMATCH'}")

    return npz_path


if __name__ == "__main__":
    print("=" * 50)
    print("Keras Weights -> NumPy Export")
    print("=" * 50)

    exported = []
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        npz_path = export_weights(name, path)
        exported.append(npz_path)
        print()

    print(f"Done. Exported {len(exported)} models.")
    if exported:
        print("Files to commit:")
        for p in exported:
            print(f"  {os.path.basename(p)}")
