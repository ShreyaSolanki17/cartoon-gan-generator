"""Smoke test for the FastAPI backend (api.py). Plain asserts, no test framework.

Bypasses the lifespan startup (which needs a real checkpoint on disk) by
injecting a throwaway randomly-initialized generator directly.

Run: python test_api.py
"""
import base64
import io

from PIL import Image

import api
import generate
from model import CartoonGenerator


def test_generate_endpoint():
    api.gen = CartoonGenerator().to(generate.DEVICE)
    result = api.generate(api.GenerateRequest(attrs=api.FIXED_ATTRS))
    assert set(result.keys()) == {"color", "face_outline", "hair_outline"}
    for key, b64 in result.items():
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert img.size == (128, 128), (key, img.size)
    print("test_generate_endpoint passed")


def test_generate_rejects_out_of_range_attrs():
    api.gen = CartoonGenerator().to(generate.DEVICE).to(api.DEVICE) if hasattr(api, "DEVICE") else CartoonGenerator()
    bad_attrs = list(api.FIXED_ATTRS)
    bad_attrs[0] = 999
    try:
        api.generate(api.GenerateRequest(attrs=bad_attrs))
        assert False, "expected HTTPException"
    except api.HTTPException as e:
        assert e.status_code == 400
    print("test_generate_rejects_out_of_range_attrs passed")


if __name__ == "__main__":
    test_generate_endpoint()
    test_generate_rejects_out_of_range_attrs()
    print("All smoke tests passed.")
