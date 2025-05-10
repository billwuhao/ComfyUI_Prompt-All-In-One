import os
from PIL import Image
import io
from io import BytesIO
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import torch
from google import genai
from google.genai import types


def imgtensor_to_bytes(tensor):
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    img_array = tensor.cpu().squeeze().to(torch.uint8).numpy()
    img = Image.fromarray(img_array, mode='RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")

    return buffered.getvalue()


def audio_tensor_to_mp3_bytes(audio_tensor, sample_rate=44100):
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=0)  # 取平均转为单通道
    # 转换为 numpy 数组并调整范围
    audio_data = audio_tensor.cpu().numpy()
    if audio_data.max() <= 1.0:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    # 创建 WAV 缓冲区
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_data)
    
    # 转换为 MP3
    audio = AudioSegment.from_wav(wav_buffer)
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3")

    return mp3_buffer.getvalue()


def get_config(temperature, top_p, top_k, max_output_tokens, seed):
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        max_output_tokens=max_output_tokens,
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_NONE',
            )
        ],
    )
    return config


class APIGeminiTextGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "gemini-2.0-flash-lite",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash-8b",
                    ],
                    {"default": "gemini-2.0-flash-lite"},
                ),
                "proxy": ("STRING", {"default": "http://127.0.0.1:None", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 0, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        prompt,
        model,
        proxy,
        temperature, 
        top_p, 
        top_k, 
        max_output_tokens,
        seed,
    ):
        if proxy.strip() != "" or proxy.strip() != "http://127.0.0.1:None":
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            os.environ['NO_PROXY'] = "localhost,127.0.0.1"

        if os.getenv("GOOGLE_API_KEY") is not None:
            API_KEY = os.getenv("GOOGLE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = genai.Client(
            api_key=API_KEY,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=get_config(temperature, top_p, top_k, max_output_tokens, seed),
        )

        return (response.text,)


class APIGeminiImgOrAudioOrVideo2Text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "gemini-2.5-pro-exp-03-25",
                        "gemini-2.5-flash-preview-04-17",
                        "gemini-2.5-pro-preview-05-06",
                        "gemini-2.0-flash",
                        "gemini-2.0-flash-exp-image-generation",
                        "gemini-2.0-flash-thinking-exp-01-21",
                        "gemini-1.5-flash",
                    ],
                    {"default": "gemini-2.5-pro-preview-05-06"},
                ),
                "proxy": ("STRING", {"default": "http://127.0.0.1:None", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 0, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        prompt,
        model,
        proxy,
        temperature, 
        top_p, 
        top_k, 
        max_output_tokens,
        seed,
        image=None,
        audio=None,
        video=None,
    ):
        if proxy.strip() != "" or proxy.strip() != "http://127.0.0.1:None":
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            os.environ['NO_PROXY'] = "localhost,127.0.0.1"

        if os.getenv("GOOGLE_API_KEY") is not None:
            API_KEY = os.getenv("GOOGLE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = genai.Client(
            api_key=API_KEY,
        )

        contents = [{"parts": [{"text": prompt},]}]

        if video is not None:
            parts = []
            for img in video:
                parts.append(
                    {"inline_data": {
                            "mime_type": "image/png",
                            "data": imgtensor_to_bytes(img),
                        }
                    }
                )
            contents[0]["parts"].extend(parts)

        elif image is not None:
            part = {"inline_data": {
                        "mime_type": "image/png",
                        "data": imgtensor_to_bytes(image),
                    }
                }
            contents[0]["parts"].append(part)
        
        if audio is not None:
            audio_part = {"inline_data": {
                        "mime_type": "audio/mp3",
                        "data": audio_tensor_to_mp3_bytes(
                                audio["waveform"].squeeze(0),
                                audio["sample_rate"],),
                    }
                }
            contents[0]["parts"].append(audio_part)

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=get_config(temperature, top_p, top_k, max_output_tokens, seed),
        )

        return (response.text,)


class APIGeminiImageGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "gemini-2.0-flash-exp-image-generation",
                        "gemini-2.0-flash-preview-image-generation",
                    ],
                    {"default": "gemini-2.0-flash-exp-image-generation"},
                ),
                "proxy": ("STRING", {"default": "http://127.0.0.1:None", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 0, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        prompt,
        model,
        proxy,
        temperature, 
        top_p, 
        top_k, 
        max_output_tokens,
        seed,
        image=None,
    ):
        if proxy.strip() != "" or proxy.strip() != "http://127.0.0.1:None":
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            os.environ['NO_PROXY'] = "localhost,127.0.0.1"

        if os.getenv("GOOGLE_API_KEY") is not None:
            API_KEY = os.getenv("GOOGLE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = genai.Client(
            api_key=API_KEY,
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_output_tokens=max_output_tokens,
            safety_settings=[
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE',
                )
            ],
            response_modalities=['Text', 'Image']  # Critical for image generation
        )
        if image is not None:
            img_tensor = image.cpu()
            image_arr = img_tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
            img = Image.fromarray(image_arr, mode='RGB')
            contents = [prompt, img]
        else:
            contents = [prompt]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        response_text = ""
        image_bytes = None

        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text + "\n"
                        if hasattr(part, 'begin') and part.begin:
                            response_text += part.begin

                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_bytes = part.inline_data.data

        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_arr = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_arr)[None,]
        else:
            img_tensor = torch.zeros((1, 512, 512, 3))

        return img_tensor, response_text