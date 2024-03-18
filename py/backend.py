MODEL = "SimianLuo/LCM_Dreamshaper_v7"
SCHEDULER = "LCMScheduler"
LORA = "latent-consistency/lcm-lora-sdv1-5"
STEPS = 4
PROMPT = "best quality, amazing quality, very aesthetic, absurdres"
NEGATIVE_PROMPT = "lowres, bad, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, abstract"
SEED = 233
WARMUPS = 3
BATCH = 1
HEIGHT = 768
WIDTH = 512
GUIDANCE_SCALE = 1.0


import base64
import argparse
import importlib
from io import BytesIO
import os
import signal
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

import torch
from diffusers import AutoPipelineForText2Image
from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=4201)
args = parser.parse_args()


def load_model(model):
    model = AutoPipelineForText2Image.from_pretrained(
        model,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    scheduler_cls = getattr(importlib.import_module("diffusers"), SCHEDULER)
    model.scheduler = scheduler_cls.from_config(model.scheduler.config)
    # model.load_lora_weights(LORA)
    # model.fuse_lora()
    model.to(torch.device("cuda"))
    return model


def compile_model(model):
    import xformers
    import triton

    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    model = compile(model, config)
    return model


model = load_model(MODEL)
model = compile_model(model)


app = FastAPI()


class Request(BaseModel):
    prompt: str
    seed: Optional[int] = Field(default=None)
    np: str = Field(default=NEGATIVE_PROMPT)
    cfg_scale: float = Field(default=0, ge=0, le=10)
    width: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    height: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    steps: int = Field(default=4, ge=1, le=8)
    batch: int = Field(default=1, ge=0, le=4)

    @validator("cfg_scale")
    def scale(cls, v):
        # return (v / 10 if v else v) + 1.0 # LCM lora
        return v + 3  # LCM pipeline 3-13


class Response(BaseModel):
    images: list[str]


ERROR_COUNT = 0


@app.post("/t2i")
async def t2i(data: Request) -> Response:
    try:
        generator = (
            None
            if data.seed is None
            else torch.Generator(device="cuda").manual_seed(data.seed)
        )
        kwarg_inputs = dict(
            prompt=data.prompt + ", " + PROMPT,
            # negative_prompt=data.np,
            height=data.height,
            width=data.width,
            num_inference_steps=data.steps,
            num_images_per_prompt=data.batch,
            generator=generator,
            guidance_scale=data.cfg_scale,
        )
        print(kwarg_inputs)
        images = model(**kwarg_inputs).images
        result = []
        for image in images:
            output_buffer = BytesIO()
            image.save(output_buffer, format="PNG")
            base64_image = base64.b64encode(output_buffer.getvalue())
            result.append(base64_image)

        return Response(images=result)
    except Exception as e:
        global ERROR_COUNT
        ERROR_COUNT += 1
        print(e)
        print(f"Error count: {ERROR_COUNT}")
        if ERROR_COUNT > 3:
            os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
