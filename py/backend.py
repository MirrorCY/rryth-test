STEPS = 4
PROMPT = "best quality, amazing quality, very aesthetic, absurdres"
NEGATIVE_PROMPT = "lowres, bad, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, abstract"
SEED = 233
WARMUPS = 3
BATCH = 1
HEIGHT = 768
WIDTH = 512
GUIDANCE_SCALE = 1.0


from datetime import datetime, timedelta
from diffusers import DiffusionPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import T5EncoderModel, CLIPTextModel
from torchao.quantization import quantize_, int8_weight_only
import torch

ckpt_id = "black-forest-labs/FLUX.1-schnell"

# 单独量化每个组件。
# 如果质量受到影响，则不要量化所有组件。

############ Diffusion Transformer ############
transformer = FluxTransformer2DModel.from_pretrained(
    ckpt_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
quantize_(transformer, int8_weight_only())
print("Diffusion Transformer Done")

############ Text Encoder ############
text_encoder = CLIPTextModel.from_pretrained(
    ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
)
quantize_(text_encoder, int8_weight_only())
print("Text Encoder Done")

############ Text Encoder 2 ############
text_encoder_2 = T5EncoderModel.from_pretrained(
    ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
quantize_(text_encoder_2, int8_weight_only())
print("Text Encoder 2 Done")

############ VAE ############
vae = AutoencoderKL.from_pretrained(
    ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16
)
quantize_(vae, int8_weight_only())
print("VAE Done")

pipeline = DiffusionPipeline.from_pretrained(
    ckpt_id,
    transformer=transformer,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    torch_dtype=torch.bfloat16,
).to("cuda")

# pipeline.load_lora_weights("./flux_dev_softstyle_araminta_k.safetensors")
# pipeline.load_lora_weights("./flat_colour_anime_style_schnell_v3.4.safetensors")

import base64
import argparse
from datetime import datetime, timedelta
from io import BytesIO
import os
import signal
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from collections import deque
import hashlib


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=4201)
args = parser.parse_args()

app = FastAPI()


class Request(BaseModel):
    prompt: str
    seed: Optional[int] = Field(default=None)
    np: str = Field(default=NEGATIVE_PROMPT)
    cfg_scale: float = Field(default=0, ge=0, le=10)
    width: int = Field(default=512, ge=64, le=1344, multiple_of=64)
    height: int = Field(default=512, ge=64, le=1344, multiple_of=64)
    steps: int = Field(default=4, ge=1, le=8)
    batch: int = Field(default=1, ge=0, le=4)

    @validator("cfg_scale")
    def scale(cls, v):
        # return (v / 10 if v else v) + 1.0 # LCM lora
        # return v + 3  # LCM pipeline 3-13
        return v  # Flux


class Response(BaseModel):
    images: list[str]


ERROR_COUNT = 0
REQUEST_RECORDS = deque(maxlen=10)
REQUEST_TIMEOUT = timedelta(seconds=10)


@app.post("/t2i")
async def t2i(data: Request) -> Response:
    global REQUEST_RECORDS
    request_hash = hashlib.md5(data.model_dump_json().encode()).hexdigest()
    current_time = datetime.now()

    for record in REQUEST_RECORDS:
        if (
            record["hash"] == request_hash
            and (current_time - record["timestamp"]) < REQUEST_TIMEOUT
        ):
            raise HTTPException(status_code=429)
    REQUEST_RECORDS.append({"hash": request_hash, "timestamp": current_time})

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
            num_inference_steps=4,  # data.steps,
            num_images_per_prompt=1,  # data.batch,
            generator=generator,
            guidance_scale=data.cfg_scale,
        )
        print(kwarg_inputs)
        images = pipeline(**kwarg_inputs).images
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
