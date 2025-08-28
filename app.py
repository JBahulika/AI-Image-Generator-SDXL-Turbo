# ==============================================================================
# IMPORTS AND INITIAL SETUP
# ==============================================================================
import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import time

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
# Set up the page with a modern title, icon, and layout.
st.set_page_config(
    page_title="AI Image Generator | SDXL Turbo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MODEL LOADING (CACHED)
# ==============================================================================
# Description: This function loads the SDXL Turbo model and is cached to prevent
# reloading every time a widget is changed. This is crucial for performance.
# ------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Loads the SDXL Turbo pipeline and moves it to the appropriate device."""
    print("🚀 Initializing AI model... (This will run only once)")
    
    model_id = "stabilityai/sdxl-turbo"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float32 # float32 is more stable for MPS

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True
    )
    
    pipe = pipe.to(device)
    
    # Enable memory-saving offloading for MPS devices
    if device == "mps":
        pipe.enable_model_cpu_offload()
        
    print("✅ AI Model ready.")
    return pipe

# ==============================================================================
# IMAGE GENERATION FUNCTION
# ==============================================================================
# Description: This function takes the pipeline and all user-defined parameters
# to generate a new image.
# ------------------------------------------------------------------------------
def generate_image(pipe, prompt, negative_prompt, height, width, steps, guidance, seed):
    """Generates an image based on the provided parameters."""
    
    # Use a CPU generator for reproducibility on MPS devices
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    start_time = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    end_time = time.time()
    
    generation_time = end_time - start_time
    return image, generation_time

# ==============================================================================
# STREAMLIT UI LAYOUT
# ==============================================================================

# --- Header ---
st.title("🎨 AI Image Generator")
st.markdown("Create stunning images from text with the power of **SDXL Turbo**. Adjust the settings in the sidebar to craft your perfect masterpiece.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🛠️ Generation Controls")
    st.markdown("Fine-tune the AI's creative process.")

    # --- Main Prompts ---
    prompt = st.text_area(
        "**Enter your main prompt**",
        "A cinematic shot of a raccoon wearing a tiny top hat, sitting at a cafe table, rainy night, neon lights, hyperrealistic.",
        height=150
    )
    negative_prompt = st.text_area(
        "**Enter a negative prompt** (what to avoid)",
        "blurry, ugly, deformed, text, watermark, signature",
        height=100
    )

    st.divider()

    # --- Image Dimensions ---
    st.subheader("Image Dimensions")
    # Using a select slider for specific, tested resolutions
    resolution_options = {
        "Square (512x512)": (512, 512),
        "Portrait (512x768)": (512, 768),
        "Landscape (768x512)": (768, 512),
        "HD Square (768x768)": (768, 768)
    }
    selected_resolution = st.select_slider(
        "Select a resolution",
        options=list(resolution_options.keys()),
        value="HD Square (768x768)"
    )
    height, width = resolution_options[selected_resolution]

    st.divider()

    # --- Advanced Settings (in an expander) ---
    with st.expander("Advanced Settings"):
        steps = st.slider("Inference Steps", 1, 20, 8, help="Number of steps for the model. SDXL Turbo works well with few steps (5-10).")
        guidance = st.slider("Guidance Scale (CFG)", 0.0, 5.0, 1.0, 0.5, help="How strictly the AI should follow the prompt. For Turbo models, this should be very low.")
        seed = st.number_input("Seed", value=42, help="A specific seed ensures you get the same image every time for the same prompt.")

# --- Main Content Area ---
col1, col2 = st.columns([0.6, 0.4])

with col1:
    # The "Generate" button is the main trigger
    if st.button("✨ Generate Image", use_container_width=True, type="primary"):
        # Load the model (will be cached after the first run)
        with st.spinner("Warming up the AI... This may take a moment on first run."):
            pipeline = load_pipeline()

        # Generate the image with a new spinner
        with st.spinner("Creating your masterpiece... 🎨"):
            try:
                image, gen_time = generate_image(pipeline, prompt, negative_prompt, height, width, steps, guidance, seed)
                
                # Store the generated image in session state to persist it
                st.session_state.generated_image = image
                st.session_state.generation_time = gen_time
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

with col2:
    st.subheader("Generated Output")
    # Display the image if it exists in the session state
    if "generated_image" in st.session_state:
        st.image(st.session_state.generated_image, caption="Your generated image.")
        
        # Display performance metric
        st.success(f"Generated in {st.session_state.generation_time:.2f} seconds.")
        
        # Prepare image for download
        from io import BytesIO
        buf = BytesIO()
        st.session_state.generated_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Add a download button
        st.download_button(
            label="💾 Download Image",
            data=byte_im,
            file_name="generated_image.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("Your generated image will appear here once you click the 'Generate' button.")

