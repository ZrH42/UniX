"""
Gradio UI for UniX Image Understanding/Generation Demo.

Usage:
    python demo_gradio.py
"""

import json
import os
import random
import numpy as np
import torch
from PIL import Image
import gradio as gr

from inference.pipeline import setup_model, DEFAULT_MODEL_PATH, DEFAULT_VAE_PATH
from inference.config import ModelConfig

# Configuration file path
CONFIG_FILE = "demo_gradio_config.json"

# Image size required by the model
MODEL_IMAGE_SIZE = 384

# Demo images directory
DEMO_IMAGES_DIR = "inference/demo_image"


UNDERSTANDING_PROMPTS = [
    "As an imaging expert, review this X-ray and share the FINDINGS and IMPRESSION.",
    "Review this X-ray and provide a detailed FINDINGS and IMPRESSION.",
    "Provide the FINDINGS and IMPRESSION for this chest X-ray.",
]

GENERATION_PROMPTS = [
    "No focal consolidation, pleural effusion, pneumothorax, or pulmonary edema is detected. Heart and mediastinal contours are within normal limits.",
    "Feeding tube ends low in the stomach. Moderate left lower lobe atelectasis. Small left pleural effusion has decreased. No pneumothorax. Right lung clear. Normal postoperative appearance to cardiomediastinal silhouette.",
    "There may have been a modest decrease in small residual right pleural effusion. Right basal atelectasis is still substantial. No pneumothorax. Left lung clear. Heart size top normal, unchanged. Left PIC line ends low in the SVC.",
]

# Global model storage
_model_inferencer = None
_model_loaded = False


def load_config():
    """Load configuration from file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def resize_image_for_model(image):
    """Resize image to model's required size (384x384)."""
    return image.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.LANCZOS)


def get_demo_images():
    """Get list of demo images from the demo directory."""
    if not os.path.exists(DEMO_IMAGES_DIR):
        return []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    return sorted([
        f for f in os.listdir(DEMO_IMAGES_DIR)
        if f.lower().endswith(supported_formats)
    ])


def load_and_cache_model(model_path, vae_path):
    """Load model and return inferencer."""
    global _model_inferencer, _model_loaded
    config = ModelConfig(
        model_path=model_path,
        vae_path=vae_path,
    )
    _model_inferencer = setup_model(config, load_vae=True)
    _model_loaded = True
    return _model_inferencer


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_demo_image(selected_name):
    """Load demo image when selected from dropdown."""
    if not selected_name or selected_name == "No demo images available":
        return None
    demo_path = os.path.join(DEMO_IMAGES_DIR, selected_name)
    if os.path.exists(demo_path):
        return Image.open(demo_path)
    return None


def on_prompt_select_change(selected):
    """Show/hide custom prompt textbox based on selection."""
    return gr.update(visible=(selected == "Custom prompt..."))


def handle_understanding(uploaded_image, prompt_select, custom_prompt, chatbot_value):
    """Handle image understanding task."""
    global _model_inferencer, _model_loaded

    if uploaded_image is None:
        return chatbot_value, None, "Please upload an image first."

    if prompt_select == "Custom prompt...":
        prompt = custom_prompt
    else:
        prompt = prompt_select

    if not prompt or not prompt.strip():
        return chatbot_value, uploaded_image, "Please enter a prompt."

    resized_image = resize_image_for_model(uploaded_image)

    if not _model_loaded or _model_inferencer is None:
        return chatbot_value, uploaded_image, "Model not loaded. Please load model from sidebar."

    # Build chatbot history as list of tuples
    history = chatbot_value if chatbot_value else []
    history.append([prompt, ""])

    try:
        output = _model_inferencer(
            image=resized_image,
            text=prompt,
            understanding_output=True
        )
        response = output.get("text", "No response generated.")
    except Exception as e:
        response = f"Error during inference: {str(e)}"

    history[-1][1] = response

    return history, uploaded_image, ""


def handle_generation(prompt_select, custom_prompt, cfg_scale, seed, chatbot_value):
    """Handle image generation task."""
    global _model_inferencer, _model_loaded

    if prompt_select == "Custom prompt...":
        prompt = custom_prompt
    else:
        prompt = prompt_select

    if not prompt or not prompt.strip():
        return chatbot_value, None, "Please enter a prompt."

    if not _model_loaded or _model_inferencer is None:
        return chatbot_value, None, "Model not loaded. Please load model from sidebar."

    # Set seed for reproducibility
    set_seed(seed)

    # Build chatbot history
    history = chatbot_value if chatbot_value else []
    history.append([prompt, "Generating..."])

    try:
        output = _model_inferencer(
            text=prompt,
            understanding_output=False,
            cfg_text_scale=cfg_scale,
        )
        generated_image = output.get("image", None)
    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        return history, None, f"Error during generation: {str(e)}"

    if generated_image is None:
        history[-1][1] = "No image generated."
        return history, None, "No image generated."

    history[-1][1] = "[Image generated - see above]"

    return history, generated_image, ""


def clear_history_chatbot():
    """Return empty chatbot history."""
    return []


def clear_history_image():
    """Return None for image."""
    return None


def on_load_model(m_path, vae_path):
    """Load model callback."""
    global _model_inferencer, _model_loaded

    if _model_loaded and _model_inferencer is not None:
        return "Model already loaded."

    try:
        load_and_cache_model(m_path, vae_path)
        return "Model loaded successfully!"
    except Exception as e:
        _model_inferencer = None
        _model_loaded = False
        return f"Error: {str(e)}"


def create_demo():
    """Create and configure the Gradio demo."""
    set_seed(42)

    with gr.Blocks(title="UniX Image Understanding/Generation") as demo:
        gr.Markdown("<h1 style='text-align: center; font-size: 2.5em;'>🔬 UniX Image Understanding/Generation Demo</h1>")

        with gr.Row():
            # Left sidebar - Model Configuration
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("## Model Configuration")

                saved_config = load_config()

                gr.Markdown("**Model Paths**")
                model_path = gr.Textbox(
                    label="UniX Model Path",
                    value=saved_config.get('model_path', DEFAULT_MODEL_PATH),
                    interactive=True,
                    placeholder="Path to UniX directory containing all components"
                )
                vae_path = gr.Textbox(
                    label="VAE Path (optional)",
                    value=saved_config.get('vae_path', DEFAULT_VAE_PATH),
                    interactive=True,
                    placeholder="Defaults to model_path/vae/"
                )

                def save_config_handler(m_path, vae_path):
                    save_config({
                        'model_path': m_path,
                        'vae_path': vae_path,
                    })
                    return "Configuration saved!"

                save_btn = gr.Button("Save Configuration", variant="secondary")
                save_status = gr.Markdown("")
                save_btn.click(
                    fn=save_config_handler,
                    inputs=[model_path, vae_path],
                    outputs=[save_status]
                )

                gr.Markdown("---")
                gr.Markdown("**Load Model**")

                model_status = gr.Markdown(value="**Model Status:** Not loaded")

                load_btn = gr.Button("Load Model", variant="primary")
                loading_status = gr.Markdown(value="", visible=False)

                def set_loading():
                    return gr.update(value="**Loading model...**", visible=True)

                def hide_loading():
                    return gr.update(visible=False)

                load_btn.click(
                    fn=set_loading,
                    outputs=[loading_status]
                ).then(
                    fn=on_load_model,
                    inputs=[model_path, vae_path],
                    outputs=[model_status]
                ).then(
                    fn=hide_loading,
                    outputs=[loading_status]
                )

                gr.Markdown("---")
                clear_btn = gr.Button("Clear Conversation History", variant="stop")

            # Right main area
            with gr.Column(scale=3):
                with gr.Tabs() as main_tabs:
                    # Image Understanding Tab
                    with gr.Tab("Image Understanding"):
                        gr.Markdown("### Image Understanding")

                        demo_images = get_demo_images()
                        demo_image_select = gr.Dropdown(
                            choices=["Select demo image..."] + demo_images if demo_images else ["No demo images available"],
                            value="Select demo image...",
                            label="Demo Images",
                            interactive=True
                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("")
                            with gr.Column(scale=2):
                                uploaded_image_und = gr.Image(
                                    type="pil",
                                    label="Input Image",
                                    interactive=True,
                                    height=300
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("")

                        demo_image_select.select(
                            fn=select_demo_image,
                            inputs=[demo_image_select],
                            outputs=[uploaded_image_und],
                        )

                        prompt_select_und = gr.Dropdown(
                            choices=["Custom prompt..."] + UNDERSTANDING_PROMPTS,
                            value=UNDERSTANDING_PROMPTS[0] if UNDERSTANDING_PROMPTS else "Custom prompt...",
                            label="Select Prompt",
                            interactive=True
                        )

                        custom_prompt_und = gr.Textbox(
                            label="Custom Prompt",
                            placeholder="Enter your custom prompt here...",
                            lines=3,
                            visible=False
                        )

                        prompt_select_und.change(
                            fn=on_prompt_select_change,
                            inputs=[prompt_select_und],
                            outputs=[custom_prompt_und],
                        )

                        analyze_btn = gr.Button("Analyze Image", variant="primary")

                        chatbot_und = gr.Chatbot(
                            label="Conversation History",
                            height=300,
                        )

                        status_msg_und = gr.Markdown("")

                        analyze_btn.click(
                            fn=handle_understanding,
                            inputs=[
                                uploaded_image_und,
                                prompt_select_und,
                                custom_prompt_und,
                                chatbot_und,
                            ],
                            outputs=[
                                chatbot_und,
                                uploaded_image_und,
                                status_msg_und,
                            ],
                            show_progress=True
                        )

                        clear_btn.click(
                            fn=clear_history_chatbot,
                            outputs=[chatbot_und]
                        )
                        clear_btn.click(
                            fn=clear_history_image,
                            outputs=[uploaded_image_und]
                        )

                    # Image Generation Tab
                    with gr.Tab("Image Generation"):
                        gr.Markdown("### Image Generation")

                        with gr.Row():
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                value=2.0,
                                step=0.1,
                                label="CFG Text Scale"
                            )
                            seed = gr.Number(
                                value=42,
                                label="Seed",
                                precision=0
                            )

                        prompt_select_gen = gr.Dropdown(
                            choices=["Custom prompt..."] + GENERATION_PROMPTS,
                            value="Custom prompt...",
                            label="Select Prompt",
                            interactive=True
                        )

                        custom_prompt_gen = gr.Textbox(
                            label="Custom Prompt",
                            placeholder="Enter your prompt for image generation...",
                            lines=3,
                            visible=True
                        )

                        prompt_select_gen.change(
                            fn=on_prompt_select_change,
                            inputs=[prompt_select_gen],
                            outputs=[custom_prompt_gen],
                        )

                        generate_btn = gr.Button("Generate Image", variant="primary")

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("")
                            with gr.Column(scale=2):
                                output_image_gen = gr.Image(
                                    label="Generated Image",
                                    type="pil",
                                    interactive=False,
                                    height=400
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("")

                        chatbot_gen = gr.Chatbot(
                            label="Generation History",
                            height=300,
                        )

                        status_msg_gen = gr.Markdown("")

                        generate_btn.click(
                            fn=handle_generation,
                            inputs=[
                                prompt_select_gen,
                                custom_prompt_gen,
                                cfg_scale,
                                seed,
                                chatbot_gen,
                            ],
                            outputs=[
                                chatbot_gen,
                                output_image_gen,
                                status_msg_gen,
                            ],
                            show_progress=True
                        )

                        clear_btn.click(
                            fn=clear_history_chatbot,
                            outputs=[chatbot_gen]
                        )
                        clear_btn.click(
                            fn=clear_history_image,
                            outputs=[output_image_gen]
                        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0")