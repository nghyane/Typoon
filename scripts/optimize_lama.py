#!/usr/bin/env python3
"""Optimize LaMa ONNX model: BN→Mul+Add fusion → simplify → FP16.

LaMa's FFC architecture has BN after Add (not directly after Conv),
so classical BN-into-Conv folding doesn't apply. Instead we replace
each BN with precomputed Mul(scale) + Add(bias), eliminating the
running_var/running_mean (which overflow in FP16).

Usage:
    python scripts/optimize_lama.py models/lama_inpaint.onnx models/lama_inpaint_opt.onnx
"""

import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper


def replace_bn_with_mul_add(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace BatchNormalization with precomputed Mul + Add.

    BN formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
              = x * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var + eps))
              = x * scale + bias
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    nodes_to_remove = []
    nodes_to_add = []
    inits_to_remove = set()
    inits_to_add = []

    for bn_node in graph.node:
        if bn_node.op_type != "BatchNormalization":
            continue
        if len(bn_node.input) < 5:
            continue

        # Load BN params in float64 for precision
        gamma = numpy_helper.to_array(initializers[bn_node.input[1]]).astype(np.float64)
        beta = numpy_helper.to_array(initializers[bn_node.input[2]]).astype(np.float64)
        mean = numpy_helper.to_array(initializers[bn_node.input[3]]).astype(np.float64)
        var = numpy_helper.to_array(initializers[bn_node.input[4]]).astype(np.float64)

        eps = 1e-5
        for attr in bn_node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        # Precompute: scale = gamma / sqrt(var + eps), bias = beta - mean * scale
        inv_std = 1.0 / np.sqrt(var + eps)
        scale = (gamma * inv_std).astype(np.float32)
        bias = (beta - mean * gamma * inv_std).astype(np.float32)

        # Reshape for broadcasting: [C] → [1, C, 1, 1]
        C = scale.shape[0]
        scale_4d = scale.reshape(1, C, 1, 1)
        bias_4d = bias.reshape(1, C, 1, 1)

        # Create initializers
        bn_name = bn_node.name.replace("/", "_")
        scale_name = f"{bn_name}_fused_scale"
        bias_name = f"{bn_name}_fused_bias"

        inits_to_add.append(numpy_helper.from_array(scale_4d, name=scale_name))
        inits_to_add.append(numpy_helper.from_array(bias_4d, name=bias_name))

        # Create Mul + Add nodes
        mul_out = f"{bn_node.output[0]}_mul"
        mul_node = helper.make_node(
            "Mul",
            inputs=[bn_node.input[0], scale_name],
            outputs=[mul_out],
            name=f"{bn_name}_fused_mul",
        )
        add_node = helper.make_node(
            "Add",
            inputs=[mul_out, bias_name],
            outputs=[bn_node.output[0]],
            name=f"{bn_name}_fused_add",
        )

        nodes_to_remove.append(bn_node)
        nodes_to_add.extend([mul_node, add_node])

        # Mark BN param initializers for removal
        for inp in bn_node.input[1:]:
            inits_to_remove.add(inp)

    # Apply changes
    # Insert replacement nodes at correct positions
    new_node_list = []
    for node in graph.node:
        if node in nodes_to_remove:
            # Find corresponding Mul+Add pair
            idx = nodes_to_remove.index(node)
            new_node_list.append(nodes_to_add[idx * 2])      # Mul
            new_node_list.append(nodes_to_add[idx * 2 + 1])  # Add
        else:
            new_node_list.append(node)

    del graph.node[:]
    graph.node.extend(new_node_list)

    # Update initializers
    new_inits = [i for i in graph.initializer if i.name not in inits_to_remove]
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    graph.initializer.extend(inits_to_add)

    print(f"Replaced {len(nodes_to_remove)} BN nodes with Mul+Add (precomputed scale/bias)")
    return model


def convert_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert model to FP16 using onnx's built-in float16 converter.

    Uses mixed-precision: keeps inputs/outputs as FP32, internal compute in FP16.
    The onnx converter handles type propagation and inserts Cast nodes correctly.
    """
    from onnxconverter_common import float16

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )

    # Count converted initializers
    fp16_count = sum(
        1 for init in model_fp16.graph.initializer
        if numpy_helper.to_array(init).dtype == np.float16
    )
    print(f"Converted to FP16 ({fp16_count} fp16 initializers)")
    return model_fp16


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output.onnx>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading {input_path}...")
    model = onnx.load(input_path)

    from collections import Counter
    ops_before = Counter(n.op_type for n in model.graph.node if n.op_type not in ("Constant", "Identity"))
    total_before = sum(ops_before.values())

    # Step 1: Replace BN with precomputed Mul+Add
    print("\n=== Step 1: BatchNormalization → Mul+Add ===")
    model = replace_bn_with_mul_add(model)

    # Validate
    try:
        onnx.checker.check_model(model)
        print("Model valid ✓")
    except Exception as e:
        print(f"Validation warning: {e}")

    # Step 2: Simplify (constant folding, dead code removal)
    print("\n=== Step 2: onnxsim simplification ===")
    try:
        import onnxsim
        model, check = onnxsim.simplify(model)
        print(f"Simplification {'successful ✓' if check else 'check failed ⚠'}")
    except Exception as e:
        print(f"onnxsim failed: {e}, skipping")

    # Step 3: FP16 conversion (now safe — no running_var overflow)
    print("\n=== Step 3: FP16 conversion ===")
    model = convert_to_fp16(model)

    # Summary
    ops_after = Counter(n.op_type for n in model.graph.node if n.op_type not in ("Constant", "Identity"))
    total_after = sum(ops_after.values())

    print(f"\n=== Summary ===")
    print(f"Ops: {total_before} → {total_after} (delta {total_after - total_before:+d})")
    print(f"BatchNormalization: {ops_before.get('BatchNormalization', 0)} → {ops_after.get('BatchNormalization', 0)}")

    # Verify no running_var remains
    has_var = any("running_var" in init.name for init in model.graph.initializer)
    has_mean = any("running_mean" in init.name for init in model.graph.initializer)
    print(f"running_var/mean remaining: {'YES ⚠' if has_var or has_mean else 'NONE ✓'}")

    print(f"\nSaving to {output_path}...")
    onnx.save(model, output_path)

    import os
    orig_size = os.path.getsize(input_path) / 1024 / 1024
    new_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Size: {orig_size:.1f} MB → {new_size:.1f} MB ({100 * new_size / orig_size:.0f}%)")


if __name__ == "__main__":
    main()
