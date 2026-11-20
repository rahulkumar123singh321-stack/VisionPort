
import os
from pathlib import Path
from collections import Counter

import torch
import onnx
from ultralytics import YOLO

# STEP 1: INVENTORY
print("\n=== 1. ARTEFACT INVENTORY ===")

for name in os.listdir("."):
    path = Path(name)

    if path.suffix in [".pt", ".onnx", ".engine", ".plan"]:
        print(name, f"{path.stat().st_size / 1e6:.2f} MB")

# STEP 2: LOAD YOLO CHECKPOINT
print("\n=== 2. PYTORCH CHECKPOINT ===")

weights_path = Path("yolov8n.pt")

yolo = YOLO(str(weights_path))

checkpoint = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=False
)

for key, value in checkpoint.items():
    print(key, type(value).__name__)

print(
    "Optimizer present:",
    "Yes" if checkpoint.get("optimizer") else "No"
)

# STEP 3: ARCHITECTURE
print("\n=== 3. MODEL ARCHITECTURE ===")

model = yolo.model

print("Model type:", type(model).__name__)
print("Top-level layers:", len(model.model))

for i in [0, 1, 2, len(model.model) - 1]:
    layer = model.model[i]

    params = sum(
        p.numel() for p in layer.parameters()
    )

    print(
        f"Layer {i}: {type(layer).__name__}, "
        f"{params:,} parameters"
    )

# STEP 4: WEIGHTS
print("\n=== 4. MODEL WEIGHTS ===")

state = model.state_dict()

total_params = sum(
    p.numel() for p in model.parameters()
)

first = next(model.parameters())

print("Named tensors:", len(state))
print("Total parameters:", f"{total_params:,}")
print("Loaded precision:", first.dtype)

print(
    "Approximate memory:",
    f"{total_params * first.element_size() / 1e6:.2f} MB"
)

for name in list(state)[:3]:
    print(name, tuple(state[name].shape))

print("Number of classes:", len(model.names))
print("Example classes:", list(model.names.values())[:5])

# STEP 5: EXPORT TO ONNX
print("\n=== 5. EXPORTING TO ONNX ===")

try:
    onnx_path = yolo.export(
        format="onnx",
        imgsz=640
    )

    print("Export successful:", onnx_path)

except Exception as error:
    print("Export failed:", type(error).__name__, error)
    raise

# STEP 6: INSPECT ONNX GRAPH
print("\n=== 6. ONNX GRAPH ===")

model_proto = onnx.load(str(onnx_path))

onnx.checker.check_model(model_proto)

print("ONNX file:", onnx_path)
print("Format check: PASSED")

print(
    "File size:",
    f"{Path(onnx_path).stat().st_size / 1e6:.2f} MB"
)

print(
    "Opset version:",
    model_proto.opset_import[0].version
)

def describe(tensors):
    results = []

    for tensor in tensors:
        shape = [
            d.dim_value if d.HasField("dim_value")
            else d.dim_param
            for d in tensor.type.tensor_type.shape.dim
        ]

        results.append((tensor.name, shape))

    return results

weights = {
    w.name for w in model_proto.graph.initializer
}

inputs = [
    i for i in model_proto.graph.input
    if i.name not in weights
]

print("Inputs:", describe(inputs))
print("Outputs:", describe(model_proto.graph.output))

ops = Counter(
    node.op_type for node in model_proto.graph.node
)

print("Total operations:", len(model_proto.graph.node))
print("Weight tensors:", len(model_proto.graph.initializer))

print(
    "Most common operations:",
    ops.most_common(6)
)

print("\n=== ONNX GRAPH INSPECTION COMPLETED ===")
