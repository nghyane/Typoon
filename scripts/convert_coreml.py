#!/usr/bin/env python3
"""Convert LaMa ONNX → CoreML .mlpackage via onnx2torch + coremltools."""
import warnings
warnings.filterwarnings("ignore")

import torch, onnx, onnx2torch, coremltools as ct
import numpy as np, time, os

MODEL = "models/lama_inpaint.onnx"
OUTPUT = "models/lama_inpaint.mlpackage"

print("1/4 ONNX → PyTorch...", flush=True)
torch_model = onnx2torch.convert(onnx.load(MODEL))
torch_model.eval()

print("2/4 Tracing...", flush=True)
traced = torch.jit.trace(torch_model, (torch.randn(1,3,512,512), torch.zeros(1,1,512,512)))

print("3/4 PyTorch → CoreML MLProgram FP16...", flush=True)
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType("image", shape=(1,3,512,512)), ct.TensorType("mask", shape=(1,1,512,512))],
    outputs=[ct.TensorType("output")],
    minimum_deployment_target=ct.target.macOS14,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
)

print("4/4 Saving .mlpackage...", flush=True)
mlmodel.save(OUTPUT)

size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(OUTPUT) for f in fn)
print(f"Saved: {size/1024/1024:.1f}MB", flush=True)

# Benchmark
print("\nBenchmark:", flush=True)
img = np.random.rand(1,3,512,512).astype(np.float32)
mask = np.zeros((1,1,512,512), dtype=np.float32)
mask[0,0,100:300,100:400] = 1.0
_ = mlmodel.predict({"image": img, "mask": mask})
t0 = time.time()
for _ in range(4):
    _ = mlmodel.predict({"image": img, "mask": mask})
print(f"CoreML native: {(time.time()-t0)*250:.0f}ms/tile", flush=True)
