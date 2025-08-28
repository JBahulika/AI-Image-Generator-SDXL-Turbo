# 🎨 AI Image Generator with SDXL-Turbo

A high-performance text-to-image generator using the **SDXL-Turbo** model, featuring a Streamlit web UI and a standalone command-line script.

---

## 🚀 Live Demo

 live app here:

[![Streamlit App](https://jbahulika-ai-image-generator-sdxl-turbo-app-jufxmf.streamlit.app)


---

## ✨ Key Features

-   **High-Speed Generation**: Powered by `stabilityai/sdxl-turbo`, a model optimized for rapid generation in very few steps.
-   **Dual Interfaces**: Includes an interactive Streamlit web application (`app.py`) for easy use and a command-line script (`image_generator.py`) for direct execution
-   **Intelligent Hardware Acceleration**: Automatically detects and utilizes the best available compute device: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
-   **Memory Optimization**: Uses model CPU offloading on MPS devices to prevent out-of-memory errors and resource caching in the Streamlit app for a responsive UI after the initial load

---

## 🛠️ Technology Stack

Python, PyTorch, Hugging Face Diffusers, Transformers, Accelerate, Streamlit, Pillow

---

## 🔬 Technical Overview

This project leverages the distilled SDXL-Turbo model for rapid image synthesis. The code intelligently selects the best compute device and applies hardware-specific optimizations like `torch.float16` precision for CUDA and `model_cpu_offload` for Apple's MPS to manage memory effectively.The Streamlit app uses `@st.cache_resource` to load the model only once, ensuring a smooth user experience on subsequent generations

---

## 📂 Project Scripts

-   `app.py`: The main file for the Streamlit web application.
-   `image_generator.py`: A standalone script for direct, non-interactive image generation.
-   `requirements.txt`: A list of all necessary Python dependencies.
