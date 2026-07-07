# **[PiD-Image-Upscaler](https://huggingface.co/spaces/prithivMLmods/PiD-Image-Upscaler)**

PiD-Image-Upscaler is an experimental, advanced super-resolution and image-to-image refinement application based on the state-of-the-art **PiD (Pixel Diffusion Decoder)** framework by NVIDIA. This application couples local generative text-to-image and image-to-image pipelines with a structural 4-step student diffusion sampler to deliver clean, crisp $4\times$ image upscaling.

The workspace natively provides deep generation integration with Z-Image-Turbo (featuring real-time `madebyollin/taef1` latent streaming previews) and `black-forest-labs/FLUX.2-klein-4B` for image conditioning. It embeds a dynamic `gr.ImageSlider` view layer right into the frontend to let creators intuitively compare low-resolution baselines against raw pixel-space diffusion outputs. Fully GPU-accelerated and wrapped in a high-fidelity Orange Red interface theme, PiD-Image-Upscaler serves as a powerful standalone benchmarking sandbox for modern ultra-resolution rendering algorithms.

<img width="1732" height="1275" alt="screencapture-huggingface-co-spaces-prithivMLmods-PiD-Image-Upscaler-2026-07-07-10_42_04" src="https://github.com/user-attachments/assets/bdf4fac5-1b22-4c44-b7ed-639d1b88647a" />

### **Key Features**

* **NVIDIA PiD Denoising Decoder:** Features the core `nvidia/PiD` 4-step student sampler models (`2k` sweet-spot variant and `2kto4k` ultra-scale configuration) to execute $4\times$ super-resolution natively in pixel space.
* **Cascaded Generative Pipelines:** Supports unified **Text-to-Image** creation (via Z-Image Turbo integration), **Image-to-Image** manipulation (utilizing FLUX.2-Klein), and **Direct Image Upscaling** without needing a prior text phase.
* **Real-Time Step Previews:** Hooks directly into the denoising scheduler context loop to seamlessly stream progressive structural updates using fast, lightweight TAEF1 autoencoders.
* **Granular Layout Controls:** Provides advanced configurations allowing exact adjustments over generation resolution ($512^2$ mapping up to a massive $4096^2$ output), guidance scale metrics, and seed properties.
* **Adaptive Allocation Optimization:** Natively instruments custom dynamic patching contexts for `transformers` mask builder functions and `Gemma2Model` forward graphs to guarantee memory survivability under active ZeroGPU allocation limits.

### **Repository Structure**

```text
├── assets/
│   ├── 1BuKJZzpbEn6fn8hbuXWt.png
│   ├── Kba3-MJXkmGBwviwL5YRP.png
│   ├── lbAgWObZG-DknJHF-HmQ1.png
│   └── UsBIu4P36AXzLd-fZ3_EN.png
├── app.py
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To configure the PiD-Image-Upscaler suite locally, set up a Python environment with the compiled libraries defined below. A dedicated, high-performance CUDA-capable GPU is strictly required to execute the models.

#### **Standard PIP Installation**

**1. Install Pre-requirements**
Ensure your local system package manager is upgraded to align smoothly with modern wheel distributions:

```bash
pip install pip>=26.1

```

**2. Install Core Dependencies**
Install the primary deep learning stack, diffusion utilities, and ecosystem structures:

```bash
pip install -r requirements.txt

```

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, which guarantees rapid virtual environment synchronization and completely deterministic execution paths.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/PiD-Image-Upscaler.git
cd PiD-Image-Upscaler

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

### **Core Requirements List**

The application depends on the following libraries (defined explicitly inside `requirements.txt`):

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/diffusers.git
git+https://github.com/huggingface/peft.git
opencv-python-headless
huggingface_hub
sentencepiece 
torchvision
termcolor
loguru
omegaconf 
kernels
hydra-core
spaces
einops
fvcore
gradio==5.49.1
hf_xet
torch==2.11.0
numpy
iopath 
imageio
boto3
botocore
pyyaml
av
wandb

```

### **Usage**

Once the FastAPI web deployment initializes, load the dashboard by pointing your browser to the local loopback endpoint (typically `http://127.0.0.1:7860/`).

1. **Image2ImagePiD Tab:** Upload an image file, type a modification prompt context, configure guidance parameters, and click **Run**. The slider block will display the FLUX.2-Klein modification on the **left** and the $4\times$ PiD upscale resolution on the **right**.
2. **Text2ImagePiD Tab:** Type a structural descriptive text prompt and click **Run**. Watch the live low-resolution TAEF1 preview step iterations convert into a definitive high-fidelity $4\times$ comparative layout.
3. **Image-Upscaler-(preview) Tab:** Drop any low-resolution source image asset directly onto the target pane, write an optional feature description to help guide model weight texturing, and click **Upscale 4×** to directly trigger pixel diffusion.

### **License and Source**

* **License:** [NVIDIA PiD Research License](https://github.com/PRITHIVSAKTHIUR/PiD-Image-Upscaler/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/PiD-Image-Upscaler.git](https://github.com/PRITHIVSAKTHIUR/PiD-Image-Upscaler.git)
