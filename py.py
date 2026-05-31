import json
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model('mnist_cnn.h5', compile=False)
os.makedirs('tfjs_model_fixed', exist_ok=True)

# Get model config
config = json.loads(model.to_json())

# Fix InputLayer — this is the key fix
layers = config['config']['layers']
for layer in layers:
    if layer['class_name'] == 'InputLayer':
        lc = layer['config']
        # TF.js needs 'batchInputShape' not 'batch_input_shape'
        if 'batch_input_shape' in lc:
            lc['batchInputShape'] = lc['batch_input_shape']
            del lc['batch_input_shape']

# Export weights as single shard
all_specs = []
all_data = bytearray()

for layer in model.layers:
    weights = layer.get_weights()
    if not weights:
        continue
    for i, w in enumerate(weights):
        name = 'kernel' if i == 0 else 'bias'
        flat = w.astype(np.float32).flatten().tobytes()
        all_specs.append({
            "name": f"{layer.name}/{name}",
            "shape": list(w.shape),
            "dtype": "float32"
        })
        all_data.extend(flat)

# Write single weights file
with open('tfjs_model_fixed/weights.bin', 'wb') as f:
    f.write(all_data)

# Write model.json
model_json = {
    "modelTopology": config,
    "weightsManifest": [{
        "paths": ["weights.bin"],
        "weights": all_specs
    }],
    "format": "layers-model",
    "generatedBy": "keras v2",
    "convertedBy": "manual"
}

with open('tfjs_model_fixed/model.json', 'w') as f:
    json.dump(model_json, f, indent=2)

print("Done! Files in tfjs_model_fixed/:")
for f in os.listdir('tfjs_model_fixed'):
    print(" -", f)