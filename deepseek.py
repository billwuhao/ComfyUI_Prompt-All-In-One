from openai import OpenAI
import os



class DeepSeekV3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "DeepSeek-V3",
                    ],
                    {"default": "DeepSeek-V3"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
        system_prompt,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DEEPSEEK_API_KEY") is not None:
            API_KEY = os.getenv("DEEPSEEK_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com",
        )
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}],
            stream=False
            )
            
        return (completion.choices[0].message.content,)



class DeepSeekR1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "DeepSeek-R1",
                    ],
                    {"default": "DeepSeek-R1"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DEEPSEEK_API_KEY") is not None:
            API_KEY = os.getenv("DEEPSEEK_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com",
        )
        completion = client.chat.completions.create(
            model='deepseek-reasoner',
            messages=[
                {'role': 'user', 'content': prompt}],
            stream=False
            )
            
        return (completion.choices[0].message.content, completion.choices[0].message.reasoning_content,)

