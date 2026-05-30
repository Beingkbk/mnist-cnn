# import json
# import numpy as np
# from tensorflow.keras.models import load_model

# model = load_model('mnist_cnn.h5', compile=False)

# import os
# os.makedirs('tfjs_model', exist_ok=True)

# # Export each layer's weights as .bin files
# weights_manifest = []
# all_weights = []

# for layer in model.layers:
#     weights = layer.get_weights()
#     if not weights:
#         continue
#     weight_specs = []
#     for i, w in enumerate(weights):
#         name = f"{layer.name}_{i}"
#         filename = f"weights_{name}.bin"
#         w.astype(np.float32).tofile(f"tfjs_model/{filename}")
#         weight_specs.append({
#             "name": f"{layer.name}/{['kernel','bias'][i]}",
#             "shape": list(w.shape),
#             "dtype": "float32",
#             "filename": filename
#         })
#     weights_manifest.append({
#         "layer": layer.name,
#         "weights": weight_specs
#     })

# # Save manifest
# with open('tfjs_model/weights_manifest.json', 'w') as f:
#     json.dump(weights_manifest, f, indent=2)

# print("Done! Files saved to tfjs_model/")
# print("Layers exported:", [l.name for l in model.layers if l.get_weights()])

import json
from tensorflow.keras.models import load_model

model = load_model('mnist_cnn.h5', compile=False)

# Build TF.js compatible model.json
model_json = {
    "modelTopology": json.loads(model.to_json()),
    "weightsManifest": []
}

# Load your existing weights_manifest.json
with open(r'D:\cover_letter_generator\my-portfolio\public\tfjs_model\weights_manifest.json') as f:
    manifest = json.load(f)

# Convert to TF.js weightsManifest format
weight_specs = []
for layer in manifest:
    for w in layer['weights']:
        weight_specs.append({
            "name": w['name'],
            "shape": w['shape'],
            "dtype": w['dtype']
        })

model_json['weightsManifest'] = [{
    "paths": [w['filename'] for layer in manifest for w in layer['weights']],
    "weights": weight_specs
}]

with open(r'D:\cover_letter_generator\my-portfolio\public\tfjs_model\model.json', 'w') as f:
    json.dump(model_json, f)

print("model.json created!")