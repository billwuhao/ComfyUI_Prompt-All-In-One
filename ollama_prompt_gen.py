import ollama
import base64
import io
import numpy as np
from PIL import Image


def rgba_to_rgb(image):
    """Convert RGBA image to RGB with white background"""
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
    return image

def tensor_to_pil_image(tensor):
    """Convert tensor to PIL Image with RGBA support"""
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()

    # Handle different channel counts
    if len(image_np.shape) == 2:  # Grayscale
        image_np = np.expand_dims(image_np, axis=-1)
    if image_np.shape[-1] == 1:   # Single channel
        image_np = np.repeat(image_np, 3, axis=-1)

    channels = image_np.shape[-1]
    mode = 'RGBA' if channels == 4 else 'RGB'

    image = Image.fromarray(image_np, mode=mode)
    return rgba_to_rgb(image)

def tensor_to_base64(tensor):
    """Convert tensor to base64 encoded PNG"""
    image = tensor_to_pil_image(tensor)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def sample_video_frames(video_tensor):
    """Sample frames evenly from video tensor"""
    if len(video_tensor.shape) != 4:
        raise ValueError("Video tensor must have 4 dimensions (B, C, H, W)")

    total_frames = video_tensor.shape[0]
    frames = []
    for idx in range(total_frames):
        frame = tensor_to_pil_image(video_tensor[idx])
        frames.append(frame)
    return frames


class OllamaPromptGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ([i.model for i in ollama.list().models],
                    {"default": "poluramus/llama-3.2ft_flux-prompting_v0.5:latest"},
                ),
                "max_new_tokens": ("INT", {"default": 200, "min": 0, "max": 20000}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        prompt,
        model,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        unload_model,
        image=None,
        video=None,
    ):
        
        payload = {
            "model": model,
            "prompt": prompt,
            # "images": images,
            # "stream": False,
            # "think": False,
            "keep_alive": "10m",
            # "system": "",
            "options": {
                # "num_keep": 5,
                "seed": seed,
                "num_predict": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                # "min_p": 0.0,
                # "typical_p": 0.7,
                # "repeat_last_n": 33,
                "temperature": temperature,
                # "repeat_penalty": 1.2,
                # "presence_penalty": 1.5,
                # "frequency_penalty": 1.0,
                # "penalize_newline": True,
                # "stop": ["\n", "user:"],
                # "numa": False,
                # "num_ctx": 1024,
                # "num_batch": 2,
                # "num_gpu": 1,
                # "main_gpu": 0,
                # "use_mmap": True,
                # "num_thread": 8
            }
        }

        if model == "poluramus/llama-3.2ft_flux-prompting_v0.5:latest":
            payload["prompt"] = f"Create a flux prompt from the given text: {prompt}"

        if video is not None:
            frames = sample_video_frames(video)
            frame_data = []
            for frame in frames:
                buffered = io.BytesIO()
                frame.save(buffered, format="PNG")
                frame_data.append(base64.b64encode(buffered.getvalue()).decode())
            payload["images"] = frame_data

        elif image is not None:
            payload["images"] = [tensor_to_base64(image)]

        text = ollama.generate(**payload).response

        if unload_model:
            ollama.generate(model=model, keep_alive=0)

        return (text,)