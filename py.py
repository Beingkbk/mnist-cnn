# import json
# import numpy as np
# from tensorflow.keras.models import load_model
# import os

# model = load_model('mnist_cnn.h5', compile=False)
# os.makedirs('tfjs_model_fixed', exist_ok=True)

# # Get model config
# config = json.loads(model.to_json())

# # Fix InputLayer — this is the key fix
# layers = config['config']['layers']
# for layer in layers:
#     if layer['class_name'] == 'InputLayer':
#         lc = layer['config']
#         # TF.js needs 'batchInputShape' not 'batch_input_shape'
#         if 'batch_input_shape' in lc:
#             lc['batchInputShape'] = lc['batch_input_shape']
#             del lc['batch_input_shape']

# # Export weights as single shard
# all_specs = []
# all_data = bytearray()

# for layer in model.layers:
#     weights = layer.get_weights()
#     if not weights:
#         continue
#     for i, w in enumerate(weights):
#         name = 'kernel' if i == 0 else 'bias'
#         flat = w.astype(np.float32).flatten().tobytes()
#         all_specs.append({
#             "name": f"{layer.name}/{name}",
#             "shape": list(w.shape),
#             "dtype": "float32"
#         })
#         all_data.extend(flat)

# # Write single weights file
# with open('tfjs_model_fixed/weights.bin', 'wb') as f:
#     f.write(all_data)

# # Write model.json
# model_json = {
#     "modelTopology": config,
#     "weightsManifest": [{
#         "paths": ["weights.bin"],
#         "weights": all_specs
#     }],
#     "format": "layers-model",
#     "generatedBy": "keras v2",
#     "convertedBy": "manual"
# }

# with open('tfjs_model_fixed/model.json', 'w') as f:
#     json.dump(model_json, f, indent=2)

# print("Done! Files in tfjs_model_fixed/:")
# for f in os.listdir('tfjs_model_fixed'):
#     print(" -", f)

import json
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model('mnist_cnn.h5', compile=False)
os.makedirs('tfjs_model_fixed', exist_ok=True)

# Manually build TF.js compatible config from scratch
tfjs_config = {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": [
            {
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": [None, 28, 28, 1],
                    "dtype": "float32",
                    "sparse": False,
                    "ragged": False,
                    "name": "input_layer"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "conv2d",
                    "trainable": True,
                    "filters": 32,
                    "kernel_size": [3, 3],
                    "strides": [1, 1],
                    "padding": "valid",
                    "data_format": "channels_last",
                    "dilation_rate": [1, 1],
                    "activation": "relu",
                    "use_bias": True,
                    "dtype": "float32"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "conv2d_1",
                    "trainable": True,
                    "filters": 64,
                    "kernel_size": [3, 3],
                    "strides": [1, 1],
                    "padding": "valid",
                    "data_format": "channels_last",
                    "dilation_rate": [1, 1],
                    "activation": "relu",
                    "use_bias": True,
                    "dtype": "float32"
                }
            },
            {
                "class_name": "MaxPooling2D",
                "config": {
                    "name": "max_pooling2d",
                    "trainable": True,
                    "pool_size": [2, 2],
                    "strides": [2, 2],
                    "padding": "valid",
                    "data_format": "channels_last",
                    "dtype": "float32"
                }
            },
            {
                "class_name": "Dropout",
                "config": {
                    "name": "dropout",
                    "trainable": True,
                    "rate": 0.25,
                    "dtype": "float32"
                }
            },
            {
                "class_name": "Flatten",
                "config": {
                    "name": "flatten",
                    "trainable": True,
                    "dtype": "float32"
                }
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense",
                    "trainable": True,
                    "units": 128,
                    "activation": "relu",
                    "use_bias": True,
                    "dtype": "float32"
                }
            },
            {
                "class_name": "Dropout",
                "config": {
                    "name": "dropout_1",
                    "trainable": True,
                    "rate": 0.5,
                    "dtype": "float32"
                }
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense_1",
                    "trainable": True,
                    "units": 10,
                    "activation": "softmax",
                    "use_bias": True,
                    "dtype": "float32"
                }
            }
        ]
    },
    "keras_version": "2.4.0",
    "backend": "tensorflow"
}

# Export all weights into one binary file
all_specs = []
all_data = bytearray()

for layer in model.layers:
    weights = layer.get_weights()
    if not weights:
        continue
    for i, w in enumerate(weights):
        name = 'kernel' if i == 0 else 'bias'
        all_specs.append({
            "name": f"{layer.name}/{name}",
            "shape": list(w.shape),
            "dtype": "float32"
        })
        all_data.extend(w.astype(np.float32).flatten().tobytes())

with open('tfjs_model_fixed/weights.bin', 'wb') as f:
    f.write(all_data)

model_json = {
    "modelTopology": tfjs_config,
    "weightsManifest": [{
        "paths": ["weights.bin"],
        "weights": all_specs
    }],
    "format": "layers-model",
    "generatedBy": "keras",
    "convertedBy": "manual"
}

with open('tfjs_model_fixed/model.json', 'w') as f:
    json.dump(model_json, f, indent=2)

print("Done! Files created:")
for f in os.listdir('tfjs_model_fixed'):
    print(f"  - {f}")