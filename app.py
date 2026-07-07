import os
import sys
import subprocess
import tempfile

import torch
import numpy as np
import gradio as gr
from PIL import Image
from types import SimpleNamespace
from huggingface_hub import snapshot_download

import spaces

from typing import Iterable

# --------------------------- theme ---------------------------

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red", c50="#FFF0E5", c100="#FFE0CC", c200="#FFC299", c300="#FFA366",
    c400="#FF8533", c500="#FF4500", c600="#E63E00", c700="#CC3700", c800="#B33000",
    c900="#992900", c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self, *, primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate, text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600", block_border_width="3px",
            block_shadow="*shadow_drop_lg", button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px", color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --------------------------- theme ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PID_REPO_URL = "https://github.com/nv-tlabs/PiD.git"
PID_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PiD")

if not os.path.exists(PID_REPO_DIR):
    print(f"[pid] cloning {PID_REPO_URL} -> {PID_REPO_DIR}", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", PID_REPO_URL, PID_REPO_DIR])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", PID_REPO_DIR])

os.chdir(PID_REPO_DIR)
sys.path.insert(0, PID_REPO_DIR)

snapshot_download(
    repo_id="nvidia/PiD",
    local_dir=PID_REPO_DIR,
    allow_patterns=[
        "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*",
        "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/*",
        "checkpoints/ae.safetensors",
    ],
)

from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.inference.pipeline_registry import (
    decode_with_pipeline_vae,
    extract_latent,
    load_pipeline,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

DTYPE               = torch.bfloat16
BACKBONE            = "zimage"
SR_SCALE            = 4
PID_INFERENCE_STEPS = 4
MAX_SEED            = 2**31 - 1

print("[pid] loading Z-Image pipeline...", flush=True)

from transformers import masking_utils as _mu

def _broadcasting_vmap_for_bhqkv(mask_function, bh_indices: bool = True):
    def wrapped(b, h, q, k):
        if bh_indices:
            return mask_function(
                b[:, None, None, None],
                h[None, :, None, None],
                q[None, None, :, None],
                k[None, None, None, :],
            )
        return mask_function(b, h, q[:, None], k[None, :])
    return wrapped

_mu._vmap_for_bhqkv = _broadcasting_vmap_for_bhqkv

import transformers.models.gemma2.modeling_gemma2 as _gm

_orig_gemma2_forward = _gm.Gemma2Model.forward

def _patched_gemma2_forward(self, *args, **kwargs):
    _orig_tt = torch.tensor
    dev = self.embed_tokens.weight.device
    def _tt(data, *a, **kw):
        kw.setdefault("device", dev)
        return _orig_tt(data, *a, **kw)
    torch.tensor = _tt
    try:
        return _orig_gemma2_forward(self, *args, **kwargs)
    finally:
        torch.tensor = _orig_tt

_gm.Gemma2Model.forward = _patched_gemma2_forward


pipeline, pipe_cfg = load_pipeline(BACKBONE, dtype=DTYPE)
pipeline.to("cuda")

print("[pid] loading TAEF1 (fast preview decoder)...", flush=True)
from diffusers import AutoencoderTiny

taef1 = AutoencoderTiny.from_pretrained(
    "madebyollin/taef1", torch_dtype=DTYPE, low_cpu_mem_usage=False
).to("cuda")
taef1.eval()


def _load_pid(ckpt_type: str):
    meta = get_pid_checkpoint(BACKBONE, ckpt_type)
    print(f"[pid] loading PiD decoder ({ckpt_type})...", flush=True)
    model, _ = load_model_from_checkpoint(
        experiment_name=meta.experiment,
        checkpoint_path=meta.checkpoint_path,
        config_file="pid/_src/configs/pid/config.py",
        enable_fsdp=False,
        strict=False,
    )
    model.eval()
    return model


pid_models = {
    "2k":     _load_pid("2k"),
    "2kto4k": _load_pid("2kto4k"),
}

print("[pid] loading FLUX.2-Klein pipeline...", flush=True)
from diffusers import Flux2KleinPipeline

klein_pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=DTYPE,
).to("cuda")
print("[pid] FLUX.2-Klein loaded.", flush=True)
print("[pid] ready", flush=True)

def _pick_pid_model(resolution: int):
    return pid_models["2kto4k"] if resolution > 512 else pid_models["2k"]


def _taef1_preview(packed_latent: torch.Tensor, H: int, W: int) -> Image.Image:
    with torch.no_grad():
        unpacked = extract_latent(pipeline, SimpleNamespace(images=packed_latent), pipe_cfg, H, W)
        scale = pipeline.vae.config.scaling_factor
        shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
        denorm = unpacked.to(dtype=DTYPE) / scale + shift
        img = taef1.decode(denorm).sample
        img = (img.float().clamp(-1, 1) + 1) / 2
        arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pid_pixel_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (
        (x[0].float().clamp(-1, 1) + 1) * 127.5
    ).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def _pid_stream(
    pid_model,
    latent: torch.Tensor,
    baseline_01: torch.Tensor,
    sigma: float,
    caption: str,
    num_steps: int = PID_INFERENCE_STEPS,
):
    from contextlib import nullcontext

    B = 1
    lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
    img_h, img_w = lq_h * SR_SCALE, lq_w * SR_SCALE

    caption_embs, _ = pid_model._encode_text_raw([caption])
    caption_embs = caption_embs.to(**pid_model.tensor_kwargs)

    lq_video_or_image    = (baseline_01 * 2.0 - 1.0).to(dtype=DTYPE, device="cuda")
    lq_latent            = latent.to(dtype=DTYPE, device="cuda")
    degrade_sigma_tensor = torch.tensor([sigma], device="cuda", dtype=torch.float32)

    gen   = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

    t_list = pid_model._get_t_list(device=torch.device("cuda"), num_steps=num_steps)
    autocast_ctx = (
        torch.autocast("cuda", dtype=pid_model.autocast_dtype)
        if pid_model.autocast_dtype
        else nullcontext()
    )
    net = pid_model.net
    net.eval()
    timescale           = pid_model.fm_trainer.timescale
    student_sample_type = pid_model.config.student_sample_type
    prediction_type     = pid_model.config.prediction_type

    x = noise
    with torch.no_grad(), autocast_ctx:
        steps_total = len(t_list) - 1
        for step_idx, (t_cur, t_next) in enumerate(zip(t_list[:-1], t_list[1:])):
            t_cur_batch  = t_cur.expand(B)
            t_cur_scaled = t_cur_batch * timescale
            v_pred = net(
                x,
                t_cur_scaled,
                caption_embs,
                lq_video_or_image=lq_video_or_image,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            if t_next.item() > 0:
                if student_sample_type == "ode":
                    v_for_step = pid_model._net_output_to_velocity(
                        x, v_pred, t_cur_batch, prediction_type
                    )
                    dt = t_next - t_cur
                    x  = x + dt * v_for_step
                else:
                    x0_pred   = pid_model._velocity_to_x0(x, v_pred, t_cur_batch)
                    eps_infer = torch.randn(
                        x0_pred.shape, device=x0_pred.device,
                        dtype=x0_pred.dtype, generator=gen,
                    )
                    s = [B] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.reshape(1).expand(s)
                    x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
            else:
                x = pid_model._velocity_to_x0(x, v_pred, t_cur_batch)
            yield step_idx + 1, steps_total, x.clone()


def _resize_to_divisible(image: Image.Image, max_side: int = 1024, div: int = 16) -> Image.Image:
    w, h  = image.size
    scale = min(max_side / w, max_side / h, 1.0)
    nw    = max(div, (int(w * scale) // div) * div)
    nh    = max(div, (int(h * scale) // div) * div)
    return image.resize((nw, nh), Image.LANCZOS)


def _encode_image_to_latent(image_01: torch.Tensor) -> torch.Tensor:
    vae        = pipeline.vae
    image_norm = image_01 * 2.0 - 1.0
    with torch.no_grad():
        latent = vae.encode(image_norm.to(dtype=DTYPE, device="cuda")).latent_dist.sample()
        scale  = vae.config.scaling_factor
        shift  = getattr(vae.config, "shift_factor", None) or 0.0
        latent = (latent - shift) * scale
    return latent

import random
import threading
import queue as _queue


def _generate_core(
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float   = 5.0,
    seed: int               = 0,
    resolution: int         = 512,
    randomize_seed: bool    = False,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)
    seed                = int(seed)
    num_inference_steps = int(num_inference_steps)
    H = W               = int(resolution)

    yield (
        gr.update(visible=True,  value=None, label="Generating Z-Image…"),
        gr.update(visible=False, value=None),
        gr.update(value=seed),
    )

    preview_q: "_queue.Queue" = _queue.Queue()
    _DONE = object()

    def streaming_cb(pipe, step_index, timestep, callback_kwargs):
        try:
            preview = _taef1_preview(callback_kwargs["latents"], H, W)
            preview_q.put((step_index, preview))
        except Exception as e:
            print(f"[pid] taef1 preview failed at step {step_index}: {e}", flush=True)
        return callback_kwargs

    def run_pipeline():
        gen_torch  = torch.Generator(device="cuda").manual_seed(int(seed))
        gen_kwargs = dict(
            prompt=prompt, height=H, width=W,
            num_inference_steps=num_inference_steps,
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=1,
            output_type="latent",
            generator=gen_torch,
            callback_on_step_end=streaming_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        gen_kwargs.update(pipe_cfg.extra_generate_kwargs)
        try:
            with torch.no_grad():
                out = pipeline(**gen_kwargs)
            preview_q.put((_DONE, out))
        except Exception as e:
            preview_q.put((_DONE, e))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    raw_output = None
    while True:
        step_index, payload = preview_q.get()
        if step_index is _DONE:
            if isinstance(payload, Exception):
                raise payload
            raw_output = payload
            break
        label = f"Generating Z-Image — step {step_index + 1}/{num_inference_steps}"
        yield (
            gr.update(visible=True, value=payload, label=label),
            gr.update(visible=False),
            gr.update(),
        )

    thread.join()
    final_latent = extract_latent(pipeline, raw_output, pipe_cfg, H, W)

    yield (
        gr.update(visible=True, label="Decoding final Z-Image…"),
        gr.update(visible=False),
        gr.update(),
    )
    with torch.no_grad():
        baseline_01 = decode_with_pipeline_vae(pipeline, final_latent, pipe_cfg)
    zimage_img = Image.fromarray(
        (baseline_01[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    )

    torch.cuda.empty_cache()

    final_sigma = float(pipeline.scheduler.sigmas[-1].item())
    pid_img     = None
    pid_model   = _pick_pid_model(H)

    for k, total, x in _pid_stream(pid_model, final_latent, baseline_01, final_sigma, prompt):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {k}/{total}"),
            gr.update(visible=False),
            gr.update(),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True,  value=(zimage_img, pid_img)),
        gr.update(),
    )


@spaces.GPU(duration=60)
def generate_large(*args, **kwargs):
    yield from _generate_core(*args, **kwargs)


@spaces.GPU(duration=90, size="xlarge")
def generate_xlarge(*args, **kwargs):
    yield from _generate_core(*args, **kwargs)


def generate(prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed):
    fn = generate_xlarge if int(resolution) >= 1024 else generate_large
    yield from fn(prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed)


def update_dimensions_on_upload(image: Image.Image):
    if image is None:
        return "_Upload an image to see its processed dimensions._"
    resized = _resize_to_divisible(image)
    ow, oh  = image.size
    nw, nh  = resized.size
    return (
        f"**Input:** {ow} × {oh} px  →  "
        f"**Processed:** {nw} × {nh} px  →  "
        f"**PiD output:** {nw * SR_SCALE} × {nh * SR_SCALE} px"
    )


def _i2i_generate_core(
    input_image: Image.Image,
    prompt: str,
    seed: int            = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.0,
    steps: int           = 4,
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt / description.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    input_image = _resize_to_divisible(input_image.convert("RGB"))
    W, H = input_image.size

    yield (
        gr.update(visible=True,  value=None, label="Running FLUX.2-Klein…"),
        gr.update(visible=False, value=None),
        gr.update(value=seed),
    )

    gen_torch = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        klein_out = klein_pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=gen_torch,
            output_type="pil",
        )
    klein_img: Image.Image = klein_out.images[0]

    if klein_img.size != (W, H):
        klein_img = klein_img.resize((W, H), Image.LANCZOS)

    yield (
        gr.update(visible=True, value=klein_img, label="FLUX.2-Klein done — encoding for PiD…"),
        gr.update(visible=False),
        gr.update(),
    )

    torch.cuda.empty_cache()

    klein_arr       = np.array(klein_img).astype(np.float32) / 255.0
    klein_tensor_01 = torch.from_numpy(klein_arr).permute(2, 0, 1).unsqueeze(0)

    final_latent = _encode_image_to_latent(klein_tensor_01)
    baseline_01  = klein_tensor_01.to(dtype=DTYPE, device="cuda")
    final_sigma  = float(pipeline.scheduler.sigmas[-1].item())

    pid_model = _pick_pid_model(max(H, W))
    pid_img   = None

    for k, total, x in _pid_stream(
        pid_model, final_latent, baseline_01, final_sigma, prompt,
        num_steps=PID_INFERENCE_STEPS,
    ):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {k}/{total}"),
            gr.update(visible=False),
            gr.update(),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True,  value=(klein_img, pid_img)),
        gr.update(),
    )


@spaces.GPU(duration=90, size="xlarge")
def i2i_generate(*args, **kwargs):
    yield from _i2i_generate_core(*args, **kwargs)


UPSCALER_MAX_SIDE = 1024


def _upscaler_dim_info(image: Image.Image):
    if image is None:
        return "_Upload an image to see its upscale dimensions._"
    w, h   = image.size
    scale  = min(UPSCALER_MAX_SIDE / w, UPSCALER_MAX_SIDE / h, 1.0)
    nw     = max(16, (int(w * scale) // 16) * 16)
    nh     = max(16, (int(h * scale) // 16) * 16)
    out_w, out_h = nw * SR_SCALE, nh * SR_SCALE
    return (
        f"**Input:** {w} × {h} px  →  "
        f"**Processed:** {nw} × {nh} px  →  "
        f"**Upscaled output:** {out_w} × {out_h} px  "
        f"*({SR_SCALE}× via PiD)*"
    )


def _upscaler_core(input_image: Image.Image, prompt: str):
    if input_image is None:
        raise gr.Error("Please upload an image to upscale.")

    caption = prompt.strip() if prompt and prompt.strip() else "high quality, detailed, sharp"

    img_rgb = input_image.convert("RGB")
    w, h    = img_rgb.size
    scale   = min(UPSCALER_MAX_SIDE / w, UPSCALER_MAX_SIDE / h, 1.0)
    nw      = max(16, (int(w * scale) // 16) * 16)
    nh      = max(16, (int(h * scale) // 16) * 16)
    if (nw, nh) != (w, h):
        img_rgb = img_rgb.resize((nw, nh), Image.LANCZOS)

    input_pil = img_rgb

    yield (
        gr.update(visible=True, value=input_pil, label="Encoding image…"),
        gr.update(visible=False, value=None),
    )

    arr_01    = np.array(img_rgb).astype(np.float32) / 255.0
    tensor_01 = torch.from_numpy(arr_01).permute(2, 0, 1).unsqueeze(0)

    latent      = _encode_image_to_latent(tensor_01)
    baseline_01 = tensor_01.to(dtype=DTYPE, device="cuda")
    sigma       = float(pipeline.scheduler.sigmas[-1].item())

    torch.cuda.empty_cache()

    pid_model = _pick_pid_model(max(nw, nh))
    pid_img   = None

    for k, total, x in _pid_stream(
        pid_model, latent, baseline_01, sigma, caption,
        num_steps=PID_INFERENCE_STEPS,
    ):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {k}/{total}"),
            gr.update(visible=False),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=(input_pil, pid_img)),
    )


@spaces.GPU(duration=90, size="xlarge")
def upscaler_run(*args, **kwargs):
    yield from _upscaler_core(*args, **kwargs)

DESCRIPTION = """
## PiD — Pixel Diffusion Decoder

**Text2Image** — [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) with live TAEF1 previews →
[PiD](https://huggingface.co/nvidia/PiD) 4-step pixel-diffusion 4× SR ·
**Image2Image** — FLUX.2-Klein → PiD 4× ·
**Upscaler** — PiD direct 4× ·
[GitHub ↗](https://github.com/PRITHIVSAKTHIUR/PiD-Image-Upscaler)
"""

with gr.Blocks(theme=orange_red_theme) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1 : Image2Image ──────────────────────────────────────────
        with gr.Tab("Image2Image PiD"):

            gr.Markdown(
                "Upload any image — "
                "**[FLUX.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** "
                "refines it then **PiD** super-resolves the result **4×**.  \n"
                "The slider compares the Klein output **(left)** to the PiD upscale **(right)**."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    i2i_input    = gr.Image(label="Input image", type="pil", height=340)
                    i2i_dim_info = gr.Markdown(
                        "_Upload an image to see its processed dimensions._"
                    )
                    i2i_prompt = gr.Textbox(
                        label="Prompt / description",
                        placeholder="Describe the image content or the desired style…",
                        lines=3,
                    )
                    i2i_run = gr.Button("Run Image2Image", variant="primary", size="lg")

                    with gr.Accordion("Advanced Settings", open=False):
                        i2i_seed = gr.Slider(
                            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                        )
                        i2i_rand     = gr.Checkbox(label="Randomize seed", value=True)
                        i2i_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0, maximum=10.0, step=0.1, value=1.0,
                        )
                        i2i_steps = gr.Slider(
                            label="Steps", minimum=1, maximum=50, value=4, step=1
                        )

                with gr.Column(scale=2, min_width=340):
                    i2i_live = gr.Image(
                        label="Output preview",
                        visible=True, show_label=True,
                        type="pil", height=380,
                    )
                    i2i_slider = gr.ImageSlider(
                        label="FLUX.2-Klein (left)  ↔  PiD 4× upscale (right)",
                        visible=False,
                        type="pil",
                        height=680,
                        max_height=680,
                    )

            i2i_input.upload(
                fn=update_dimensions_on_upload,
                inputs=i2i_input,
                outputs=i2i_dim_info,
            )
            i2i_run.click(
                fn=i2i_generate,
                inputs=[i2i_input, i2i_prompt, i2i_seed, i2i_rand, i2i_guidance, i2i_steps],
                outputs=[i2i_live, i2i_slider, i2i_seed],
            )

        # ── Tab 2 : Text2Image ───────────────────────────────────────────
        with gr.Tab("Text2Image PiD"):

            with gr.Row():
                prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe what you want to generate…",
                    value=(
                        "A photorealistic Labrador retriever resting beside a campfire at night, "
                        "glowing warm firelight reflecting on detailed fur, cinematic outdoor atmosphere."
                    ),
                    max_lines=1,
                    scale=4,
                    container=False,
                )
                run = gr.Button("Generate", variant="primary", scale=1)

            live_preview = gr.Image(
                label="Z-Image with PiD",
                visible=True, show_label=True,
                type="pil", height=680,
            )
            slider = gr.ImageSlider(
                label="Z-Image (left)  ↔  PiD 4× upscale (right)",
                visible=False,
                type="pil",
                height=680,
                max_height=680,
            )

            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    resolution = gr.Radio(
                        label="Z-Image resolution",
                        choices=[512, 1024],
                        value=512,
                        info="512 → 2048² (PiD 2k)  ·  1024 → 4096² (PiD 2kto4k)",
                    )
                    num_inference_steps = gr.Slider(
                        label="Z-Image steps",
                        minimum=8, maximum=50, step=1, value=28,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance",
                        minimum=1.0, maximum=10.0, step=0.5, value=5.0,
                    )
                    seed           = gr.Number(label="Seed", value=0, precision=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            run.click(
                fn=generate,
                inputs=[prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed],
                outputs=[live_preview, slider, seed],
            )

        # ── Tab 3 : Upscaler ─────────────────────────────────────────────
        with gr.Tab("Image Upscaler"):

            gr.Markdown(
                "Upload any image and **PiD** will upscale it **4×** directly — "
                "no text generation step needed.  \n"
                "An optional prompt / description helps PiD produce sharper, "
                "more faithful detail.  \n"
                "The slider compares the **original** *(left)* to the **PiD 4× upscale** *(right)*."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    up_input = gr.Image(
                        label="Image to upscale",
                        type="pil", height=360,
                    )
                    up_dim_info = gr.Markdown(
                        "_Upload an image to see its upscale dimensions._"
                    )
                    up_prompt = gr.Textbox(
                        label="Optional prompt / description",
                        placeholder="Describe the image for better detail (leave blank for auto)…",
                        lines=3,
                        visible=False,
                    )
                    up_run = gr.Button("Upscale 4x", variant="primary", size="lg")

                with gr.Column(scale=2, min_width=340):
                    up_live = gr.Image(
                        label="Output preview",
                        visible=True, show_label=True,
                        type="pil", height=380,
                    )
                    up_slider = gr.ImageSlider(
                        label="Original (left)  ↔  PiD 4× upscale (right)",
                        visible=False,
                        type="pil",
                        height=680,
                        max_height=680,
                    )

            up_input.upload(
                fn=_upscaler_dim_info,
                inputs=up_input,
                outputs=up_dim_info,
            )
            up_run.click(
                fn=upscaler_run,
                inputs=[up_input, up_prompt],
                outputs=[up_live, up_slider],
            )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True, ssr_mode=False, show_error=True)
