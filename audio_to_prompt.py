import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import tempfile
import torchaudio
from typing import Optional
import folder_paths


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "LLM")
cache_dir = folder_paths.get_temp_directory()


prompt = """Analyze the input audio and generate 6 description variants. Each variant must be <200 characters. Follow these exact definitions:

1.  `simplified`: Use only one most representative tag from the valid set.
2.  `expanded`: Broaden valid tags to include related sub-genres/techniques.
3.  `descriptive`: Convert tags into a sensory-rich sentence based *only on the sound*. DO NOT transcribe or reference the lyrics.
4.  `synonyms`: Replace tags with equivalent terms (e.g., 'strings' → 'orchestral').
5.  `use_cases`: Suggest practical applications based on audio characteristics.
6.  `analysis`: Analyze the audio's genre, instruments, tempo, and mood **based strictly on the audible musical elements**. Technical breakdown in specified format.
    *   For the `instruments` list: **Only include instruments that are actually heard playing in the audio recording.** **Explicitly ignore any instruments merely mentioned or sung about in the lyrics.** Cover all audibly present instruments.
7. `lyrical_rap_check`: if the audio is lyrical rap
**Strictly ignore any information derived solely from the lyrics when performing the analysis, especially for identifying instruments.**

**Json Output Format:**
{"simplified": <str>, "expanded": <str>, "descriptive": <str>, "synonyms": <str>, "use_cases": <str>, "analysis": { "genre": <str list>, "instruments": <str list>, "tempo": <str>, "mood": <str list>}, "lyrical_rap_check": <bool>}
"""

def find_audios(folder_path):
    """查找指定文件夹下的常见音频文件"""
    extensions = {".wav", ".MP3", ".WAV", ".mp3", ".flac", ".FLAC"}
    return [str(p.resolve()) for p in Path(folder_path).rglob("*") if p.suffix.lower() in extensions]

def cache_audio_tensor(
    cache_dir,
    audio_tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")

class MultiLinePromptKOR:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True, 
                    "default": prompt}),
                },
        }

    CATEGORY = "🎤MW/MW-Prompt-All-In-One"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "gen"
    
    def gen(self, lyrics: str):
        return (lyrics.strip(),)

MODEL_CACHE = None
PROCESSOR_CACHE = None
class KeOmniRRun:
    # def __init__(self):
    #     self.model_name = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "save_audio_prompt_to_folder": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audios_dir": ("STRING", {"default": "", "multiline": False}),
                # "tagger": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        prompt,
        max_new_tokens,
        save_audio_prompt_to_folder="",
        batch_size=1,
        audios_dir="",
        # tagger="",
        audio=None,
        seed=0,
        unload_model=False,
    ):
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        model = "Ke-Omni-R-3B"
        model_id = os.path.join(model_path, model)
        if audios_dir.strip() != "":
            audios_dir = audios_dir.strip()
            audios = find_audios(audios_dir)
            if len(audios) == 0:
                raise ValueError(f"Unable to find audio file：{audios_dir}")
        elif audio is not None:
            audio_path = cache_audio_tensor(
                cache_dir,
                audio["waveform"].squeeze(0),
                audio["sample_rate"],
            )
            audios = [audio_path]
        else:
            raise ValueError("Either audio or audios_dir must be provided.")

        audio_prompts = []
        for audio in audios:
            msg = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": audio
                        },
                        {
                            "type": "text",
                            "text": f"{prompt}\n\nOutput thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>."
                        }
                    ]
                }]
            audio_prompts.append(msg)

        global MODEL_CACHE, PROCESSOR_CACHE
        if MODEL_CACHE is None or PROCESSOR_CACHE is None:
            PROCESSOR_CACHE = Qwen2_5OmniProcessor.from_pretrained(model_id)
            MODEL_CACHE = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )

        all_outputs = []
        for i in tqdm(range(0, len(audio_prompts), batch_size)):
            batch_data = audio_prompts[i : i + batch_size]

            batch_audios, _, _ = process_mm_info(batch_data, use_audio_in_video=False)
            text = PROCESSOR_CACHE.apply_chat_template(batch_data, add_generation_prompt=True, tokenize=False)

            inputs = PROCESSOR_CACHE(
                text=text, audio=batch_audios, sampling_rate=16000, return_tensors="pt", padding=True
            ).to(MODEL_CACHE.device).to(torch.bfloat16)

            with torch.no_grad():
                generated_ids = MODEL_CACHE.generate(**inputs, max_new_tokens=max_new_tokens)

            generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
            response = PROCESSOR_CACHE.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            all_outputs.extend(response)

            print(f"Processed batch {i//batch_size + 1}/{(len(audio_prompts) + batch_size - 1)//batch_size}")

        if unload_model:
            MODEL_CACHE = None
            PROCESSOR_CACHE = None
            torch.cuda.empty_cache()

        def extract_answer(output_str):
            answer_pattern = r"<answer>(.*?)</answer>"
            match = re.search(answer_pattern, output_str)

            if match:
                return match.group(1)
            return output_str

        save_audio = save_audio_prompt_to_folder.strip()
        save_audio = save_audio if save_audio != "" else None
        
        final_output = []
        for input_index, model_output in zip(audio_prompts, all_outputs):
            model_answer = extract_answer(model_output).strip()

            # Create a result dictionary for this example
            # final_output.append({"input_index": input_index, "model_answer": model_answer, "model_output": model_output})
            final_output.append(model_answer)

            # Save results to a JSON file
            if save_audio is not None:
                output_path = input_index[0]["content"][0]["audio"].lstrip(".", 1)[0] + ".txt"
                with open(output_path, "w") as f:
                    json.dump(model_answer, f, indent=2)

        return (final_output[0],)