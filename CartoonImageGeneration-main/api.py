"""FastAPI backend for cartoon face generation. Backend only, no frontend.

Loads a trained checkpoint once at startup and serves it via one endpoint.

Run:
    CARTOON_CHECKPOINT=models/gen_final.pth uvicorn api:app
    (defaults to models/gen_final.pth if the env var is unset)

    POST /generate  body: {"attrs": [18 ints]}  (omit "attrs" to use the
    same FIXED_ATTRS sample generate.py's CLI defaults to)
    -> {"color": "<base64 png>", "face_outline": "...", "hair_outline": "..."}
"""
import base64
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generate import ATTR_MAXES, FIXED_ATTRS, load_generator, generate_image

CHECKPOINT_PATH = os.environ.get("CARTOON_CHECKPOINT", "models/gen_final.pth")

gen = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gen
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Train one with train.py, "
            "or point CARTOON_CHECKPOINT at an existing .pth file."
        )
    gen = load_generator(CHECKPOINT_PATH)
    yield


app = FastAPI(title="Cartoon Generator API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    attrs: list[int] = Field(
        default=FIXED_ATTRS, min_length=18, max_length=18,
        description=f"18 attribute indices, per-slot max: {ATTR_MAXES}",
    )


def _to_base64_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/generate")
def generate(req: GenerateRequest):
    for i, (val, max_val) in enumerate(zip(req.attrs, ATTR_MAXES)):
        if not 0 <= val <= max_val:
            raise HTTPException(400, f"attrs[{i}]={val} out of range [0, {max_val}]")

    color_img, face_outline, hair_outline = generate_image(gen, req.attrs)
    return {
        "color": _to_base64_png(color_img),
        "face_outline": _to_base64_png(face_outline),
        "hair_outline": _to_base64_png(hair_outline),
    }
