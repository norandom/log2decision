"""Describe an ONNX model: interface, operators, quantization parameters.

Usage: uv run python inspect_onnx.py <model.onnx> [--json]

`describe(path)` is also used by `export.py` to compare a trained model with the
random-weight spike model that passed the hardware gate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper


def _shape(value_info) -> list:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return dims


def _dtype(value_info) -> str:
    return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)


def describe(path: str) -> dict:
    model = onnx.load(path)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    nodes = []
    for node in graph.node:
        attrs = {}
        for a in node.attribute:
            value = onnx.helper.get_attribute_value(a)
            if isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, (list, tuple)):
                value = [v.decode() if isinstance(v, bytes) else v for v in value]
            elif isinstance(value, onnx.TensorProto):
                value = "tensor"
            attrs[a.name] = value
        nodes.append({"op": node.op_type, "domain": node.domain, "attrs": attrs,
                      "inputs": list(node.input), "outputs": list(node.output)})

    # Quantization parameters: dtype of every zero point, whether weights are symmetric,
    # whether any scale is per channel.
    zero_points = Counter()
    per_channel = 0
    weight_q = []
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        scale = inits.get(node.input[1])
        zero = inits.get(node.input[2]) if len(node.input) > 2 else None
        if scale is not None and numpy_helper.to_array(scale).size > 1:
            per_channel += 1
        if zero is not None:
            zero_points[onnx.TensorProto.DataType.Name(zero.data_type)] += 1
        if node.op_type == "DequantizeLinear" and node.input[0] in inits:
            tensor = inits[node.input[0]]
            array = numpy_helper.to_array(tensor)
            weight_q.append({
                "name": tensor.name,
                "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
                "shape": list(array.shape),
                "zero_point": None if zero is None else np.atleast_1d(numpy_helper.to_array(zero)).tolist()[:1],
                "min": int(array.min()), "max": int(array.max()),
            })

    def signature(node: dict) -> str:
        keep = {k: v for k, v in sorted(node["attrs"].items())}
        return f'{node["op"]}{json.dumps(keep, sort_keys=True)}'

    return {
        "path": path,
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opsets": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "inputs": [{"name": i.name, "dtype": _dtype(i), "shape": _shape(i)}
                   for i in graph.input if i.name not in inits],
        "outputs": [{"name": o.name, "dtype": _dtype(o), "shape": _shape(o)} for o in graph.output],
        "op_counts": dict(sorted(Counter(n["op"] for n in nodes).items())),
        "op_sequence": [n["op"] for n in nodes],
        "node_signatures": [signature(n) for n in nodes],
        "zero_point_dtypes": dict(zero_points),
        "per_channel_scales": per_channel,
        "quantized_initializers": weight_q,
        "initializer_count": len(inits),
    }


def main() -> None:
    info = describe(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(info, indent=2))
        return
    for key in ("path", "ir_version", "producer", "opsets", "inputs", "outputs", "op_counts",
                "zero_point_dtypes", "per_channel_scales", "initializer_count"):
        print(f"{key}: {info[key]}")
    print("nodes:")
    for sig in info["node_signatures"]:
        print("  ", sig)
    print("quantized initializers:")
    for w in info["quantized_initializers"]:
        print("  ", w)


if __name__ == "__main__":
    main()
