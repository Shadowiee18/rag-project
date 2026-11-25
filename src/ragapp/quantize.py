"""Make an int8 copy of the reranker, so it runs faster on the CPU (one-time step, like ingest).

Dynamic int8 quantization stores the model's weights as 8-bit integers instead of 32-bit floats: ~4x smaller,
and on CPUs with int8 instructions (AVX-512 VNNI) the matrix maths runs faster. Activations are quantized on the fly,
so no calibration data is needed. The ranking should barely change; the eval checks that.

Run:  uv run python -m ragapp.quantize     (needs the dev dependency `onnx`)
"""

import shutil
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from ragapp import config
from ragapp.rerank import int8_dir, load_model


def main() -> None:
    load_model(config.RERANK_MODEL)  # makes sure the float model is downloaded
    snapshots = config.MODEL_CACHE / f"models--{config.RERANK_MODEL.replace('/', '--')}" / "snapshots"
    source = next(p for p in snapshots.iterdir() if (p / "onnx" / "model.onnx").exists())
    target = int8_dir(config.RERANK_MODEL)
    (target / "onnx").mkdir(parents=True, exist_ok=True)
    for f in source.iterdir():  # tokenizer and config files, unchanged
        if f.is_file():
            shutil.copy(f, target / f.name)

    started = time.perf_counter()
    quantize_dynamic(source / "onnx" / "model.onnx", target / "onnx" / "model.onnx", weight_type=QuantType.QInt8)
    size = lambda p: Path(p).stat().st_size / 1e6  # noqa: E731
    print(f"{size(source / 'onnx' / 'model.onnx'):.0f} MB -> {size(target / 'onnx' / 'model.onnx'):.0f} MB "
          f"in {time.perf_counter() - started:.0f}s: {target}")


if __name__ == "__main__":
    main()
