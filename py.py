import json
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mnist_cnn.h5', compile=False)

# Get the model config and fix input layer
config = json.loads(model.to_json())

# Find and fix InputLayer config
layers = config['config']['layers']
for layer in layers:
    if layer['class_name'] == 'InputLayer':
        lc = layer['config']
        if 'batch_input_shape' in lc and 'batchInputShape' not in lc:
            lc['batchInputShape'] = lc['batch_input_shape']
        print("Fixed InputLayer:", lc)

# Build weightsManifest
weights_manifest = []
for layer in model.layers:
    ws = layer.get_weights()
    if not ws:
        continue
    paths = []
    specs = []
    for i, w in enumerate(ws):
        fname = f"weights_{layer.name}_{i}.bin"
        w.astype(np.float32).tofile(f"tfjs_model_fixed/{fname}")
        paths.append(fname)
        specs.append({
            "name": f"{layer.name}/{'kernel' if i==0 else 'bias'}",
            "shape": list(w.shape),
            "dtype": "float32"
        })
    weights_manifest.append({"paths": paths, "weights": specs})

# Write model.json
import os
os.makedirs('tfjs_model_fixed', exist_ok=True)

model_json = {
    "modelTopology": config,
    "weightsManifest": weights_manifest,
    "format": "layers-model",
    "generatedBy": "keras",
    "convertedBy": "manual"
}

with open('tfjs_model_fixed/model.json', 'w') as f:
    json.dump(model_json, f)

print("Done! Copy tfjs_model_fixed/ to your repo")
