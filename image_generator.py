# ==============================================================================
# BLOCK 1: IMPORTS AND INITIAL SETUP
# ==============================================================================
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import time

# Utility to track performance
class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"✅ Operation finished in {self.elapsed_time:.2f} seconds.")

# ==============================================================================
# BLOCK 2: CONFIGURATION AND PARAMETERS
# ==============================================================================
# Description: Centralized configuration for the new SDXL Turbo model.
# Note the significantly reduced number of inference steps.
# ------------------------------------------------------------------------------
class GenerationConfig:
    # --- Model ID ---
    # Using SDXL Turbo: a fast, single-step model perfect for memory-constrained systems.
    MODEL_ID = "stabilityai/sdxl-turbo"

    # --- Prompts ---
    PROMPT = (
        "award-winning cinematic portrait of a majestic red fox in a lush, "
        "misty forest, wearing a cozy, hand-knitted blue scarf. "
        "Golden morning light filters through the trees, dramatic lighting, "
        "hyperrealistic, ultra-detailed fur, 8k, photography."
    )
    # Negative prompts are less necessary for Turbo models but can still help.
    NEGATIVE_PROMPT = "blurry, low-resolution, ugly, deformed, disfigured."

    # --- Generation Parameters ---
    # SDXL models work best with this resolution.
    HEIGHT = 768
    WIDTH = 768
    # Turbo models require very few steps. This is the key to their speed.
    NUM_INFERENCE_STEPS = 8
    # Guidance scale should be low or 0 for Turbo models.
    GUIDANCE_SCALE = 1.0
    # Seed for reproducibility.
    SEED = 42

    # --- Output ---
    OUTPUT_FILENAME = "sdxl_turbo_generated_image.png"

# ==============================================================================
# BLOCK 3: DEVICE AND PRECISION SETUP
# ==============================================================================
# Description: Determines the best available hardware and sets the appropriate
# data type for optimal performance.
# ------------------------------------------------------------------------------
def get_compute_device():
    if torch.cuda.is_available():
        print("🚀 CUDA (NVIDIA GPU) detected. Using for acceleration.")
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        print("🚀 MPS (Apple Silicon GPU) detected. Using for acceleration.")
        return "mps", torch.float32 # Use float32 for stability on MPS
    else:
        print("⚠️ No GPU detected. Falling back to CPU (will be slow).")
        return "cpu", torch.float32

DEVICE, TORCH_DTYPE = get_compute_device()

# ==============================================================================
# BLOCK 4: MODEL LOADING AND OPTIMIZATION
# ==============================================================================
# Description: Loads the single SDXL Turbo pipeline. No refiner is needed.
# ------------------------------------------------------------------------------
def load_pipeline(config):
    print("\nLoading SDXL Turbo model... This may take a moment.")

    with Timer():
        # AutoPipelineForText2Image automatically selects the correct pipeline class.
        pipe = AutoPipelineForText2Image.from_pretrained(
            config.MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True
        )

    print("✅ Model loaded successfully.")
    print("Applying performance optimizations...")
    
    # Move the pipeline to the selected device (GPU or CPU)
    pipe = pipe.to(DEVICE)
    
    # Enable model CPU offloading for MPS to prevent memory errors.
    if DEVICE == "mps":
        print("✅ Enabling model CPU offloading for MPS.")
        pipe.enable_model_cpu_offload()

    return pipe

# ==============================================================================
# BLOCK 5: IMAGE GENERATION
# ==============================================================================
# Description: A simplified, single-stage generation process using the
# SDXL Turbo model.
# ------------------------------------------------------------------------------
def generate_image(pipe, config):
    print("\nGenerating image with SDXL Turbo...")

    # Set up a generator for reproducible results
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)

    with Timer():
        image = pipe(
            prompt=config.PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            height=config.HEIGHT,
            width=config.WIDTH,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

    return image

# ==============================================================================
# BLOCK 6: MAIN EXECUTION
# ==============================================================================
# Description: The main entry point of the script.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        config = GenerationConfig()
        pipeline = load_pipeline(config)
        final_image = generate_image(pipeline, config)
        
        # Save the final image
        final_image.save(config.OUTPUT_FILENAME)
        print(f"\n🎉 Image successfully saved as '{config.OUTPUT_FILENAME}'")

    except Exception as e:
        print(f"\n💥 An error occurred: {e}")
        print("Please ensure all dependencies are installed correctly and that your hardware has sufficient memory.")