from pathlib import Path

p = Path("scripts/export_onnx.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    "do_constant_folding=True,\n    )",
    "do_constant_folding=True,\n        dynamo=False,\n    )"
)

text = text.replace(
    "wrapper = ClassifierWrapper(model)",
    "wrapper = ClassifierWrapper(model)\n    wrapper.eval()"
)

text = text.replace(
    "wrapper = NERWrapper(model)",
    "wrapper = NERWrapper(model)\n    wrapper.eval()"
)

p.write_text(text, encoding="utf-8")

print("ONNX exporter fixed")
