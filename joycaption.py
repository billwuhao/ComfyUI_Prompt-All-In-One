
import logging
import os
from pathlib import Path
import numpy as np
import torch
import uuid
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Union
import folder_paths
import model_management


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "LLM")
device = model_management.get_torch_device()


# PIL.Image.MAX_IMAGE_PIXELS = 933120000   # Quiets Pillow from giving warnings on really large images (WARNING: Exposes a risk of DoS from malicious images)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


preset_prompts = [
    "None",
    "Write a descriptive caption for this image in a formal tone.-以正式的语气为这张图片写一个描述性的标题。",
    "Write a descriptive caption for this image in a casual tone.-以随意的语气为这张图片写一个描述性的标题。",
    "Write a stable diffusion prompt for this image.-为这张图片写一个 stable diffusion 提示。",
    "Write a MidJourney prompt for this image.-为这张图片写一个 MidJourney 提示。",
    "Write a list of Booru tags for this image.-为这张图片写一个 Booru 标签列表。",
    "Write a list of Booru-like tags for this image.-为这张图片写一个类似 Booru 的标签列表。",
    ("Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc."
     "-像艺术评论家一样分析这张图片，提供关于其构图、风格、象征意义、色彩和光线的使用、可能属于的任何艺术运动等信息。"),
    "Write a caption for this image as though it were a product listing.-为这张图片写一个标题，就像它是一个产品列表一样。",
    "Write a caption for this image as if it were being used for a social media post.-为这张图片写一个标题，就像它被用于社交媒体帖子一样。",
]

class JoyCaption:
    def __init__(self, model: str, nf4: bool):
        IS_NF4 = nf4
        # Load JoyCaption
        nf4_config = BitsAndBytesConfig(load_in_4bit=True, 
                                        bnb_4bit_quant_type="nf4", 
                                        bnb_4bit_quant_storage=torch.bfloat16,
                                        bnb_4bit_use_double_quant=True, 
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        assert isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(self.tokenizer)}"

        if IS_NF4:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(model, torch_dtype="bfloat16", quantization_config=nf4_config).eval()
            # https://github.com/fpgaminer/joycaption/issues/3#issuecomment-2619253277
            attention = self.llava_model.vision_tower.vision_model.head.attention
            attention.out_proj = torch.nn.Linear(attention.embed_dim, attention.embed_dim, device=self.llava_model.device, dtype=torch.bfloat16)

        else: 
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(model, torch_dtype="bfloat16", device_map="auto").eval()
        assert isinstance(self.llava_model, LlavaForConditionalGeneration)
    #@torch.no_grad()
    @torch.inference_mode()
    def inference(
        self,
        prompt,
        # num_workers,
        batch_size,
        max_new_tokens,
        do_sample,
        use_cache,
        temperature,
        top_k,
        top_p,
        save_img_prompt_to_folder=None,
        images_dir=None,
        tagger="",
        image=None,
        ):

        if images_dir:
            # Find the images
            image_paths = find_images(images_dir)
        elif image is not None:
            from io import BytesIO
            image = tensor2pil(image)
            if save_img_prompt_to_folder:
                image_path = os.path.join(save_img_prompt_to_folder, f"{str(uuid.uuid4())}.png")
                image.save(image_path)
                image_paths = [image_path]
            else:
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                buffer.seek(0) 
                image_paths = [buffer]
        
        tagger = tagger.strip() + ", " if tagger.strip() != "" else ""

        dataset = ImageDataset(prompt, image_paths, self.tokenizer, self.llava_model.config.image_token_index, self.llava_model.config.image_seq_length)
        dataloader = DataLoader(
                        dataset, 
                        collate_fn=dataset.collate_fn, 
                        # num_workers=num_workers, 
                        shuffle=False, 
                        drop_last=False, 
                        batch_size=batch_size
                        )
        
        end_of_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

        pbar = tqdm(total=len(image_paths), desc="Captioning images...", dynamic_ncols=True)
        
        n = 1
        for batch in dataloader:
            vision_dtype = self.llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
            vision_device = self.llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
            language_device = self.llava_model.language_model.get_input_embeddings().weight.device
            print(vision_device, vision_dtype, language_device)

            # Move to GPU
            pixel_values = batch['pixel_values'].to(vision_device, non_blocking=True)
            input_ids = batch['input_ids'].to(language_device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(language_device, non_blocking=True)

            # Normalize the image
            pixel_values = pixel_values / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(vision_dtype)

            # Generate the captions
            generate_ids = self.llava_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                suppress_tokens=None,
                use_cache=use_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Trim off the prompts
            assert isinstance(generate_ids, torch.Tensor)
            generate_ids = generate_ids.tolist()
            generate_ids = [trim_off_prompt(ids, end_of_header_id, end_of_turn_id) for ids in generate_ids]

            # Decode the captions
            captions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            captions = [c.strip() for c in captions]
            if image is not None:
                if save_img_prompt_to_folder:
                    write_caption(Path(image_paths[0]), tagger + captions[0])
                return tagger + captions[0]
            
            import shutil
            for path, caption in zip(batch['paths'], captions):
                if save_img_prompt_to_folder:
                    file_ext = os.path.splitext(path)[1]
                    img_path = os.path.join(save_img_prompt_to_folder, f"{n:0{7}d}{file_ext}")
                    shutil.copy2(path, img_path)
                    write_caption(Path(img_path), tagger + caption)
                    n += 1
                else:
                    write_caption(Path(path), tagger + caption)

            pbar.update(len(captions))

    
    def clean(self):
        import gc
        # Clean up the model
        self.llava_model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
    # Trim off the prompt
    while True:
        try:
            i = input_ids.index(eoh_id)
        except ValueError:
            break
        
        input_ids = input_ids[i + 1:]
    
    # Trim off the end
    try:
        i = input_ids.index(eot_id)
    except ValueError:
        return input_ids
    
    return input_ids[:i]


def write_caption(image_path: Path, caption: str):
    caption_path = image_path.with_suffix(".txt")
    try:
        f = os.open(caption_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)  # Write-only, create if not exist, fail if exists
    except FileExistsError:
        logging.warning(f"Caption file '{caption_path}' already exists")
        return
    except Exception as e:
        logging.error(f"Failed to open caption file '{caption_path}': {e}")
        return
    
    try:
        os.write(f, caption.encode("utf-8"))
        os.close(f)
    except Exception as e:
        logging.error(f"Failed to write caption to '{caption_path}': {e}")
        return


class ImageDataset(Dataset):
    def __init__(self, prompt: str, paths: list[Path], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], image_token_id: int, image_seq_length: int):
        self.prompt = prompt
        self.paths = paths
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]

        # Preprocess image
        # NOTE: I don't use the Processor here and instead do it manually.
        # This is because in my testing a simple resize in Pillow yields higher quality results than the Processor,
        # and the Processor had some buggy behavior on some images.
        # And yes, with the so400m model, the model expects the image to be squished into a square, not padded.
        try:
            image = Image.open(path)
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.LANCZOS)
            image = image.convert("RGB")
            pixel_values = TVF.pil_to_tensor(image)
        except Exception as e:
            logging.error(f"Failed to load image '{path}': {e}")
            pixel_values = None   # Will be filtered out later

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]

        # Format the conversation
        convo_string = self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

        # Repeat the image tokens
        input_tokens = []
        for token in convo_tokens:
            if token == self.image_token_id:
                input_tokens.extend([self.image_token_id] * self.image_seq_length)
            else:
                input_tokens.append(token)
        
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            'path': path,
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        # Filter out images that failed to load
        batch = [item for item in batch if item['pixel_values'] is not None]

        # Pad input_ids and attention_mask
        # Have to use left padding because HF's generate can't handle right padding it seems
        max_length = max(item['input_ids'].shape[0] for item in batch)
        n_pad = [max_length - item['input_ids'].shape[0] for item in batch]
        input_ids = torch.stack([torch.nn.functional.pad(item['input_ids'], (n, 0), value=self.pad_token_id) for item, n in zip(batch, n_pad)])
        attention_mask = torch.stack([torch.nn.functional.pad(item['attention_mask'], (n, 0), value=0) for item, n in zip(batch, n_pad)])

        # Stack pixel values
        pixel_values = torch.stack([item['pixel_values'] for item in batch])

        # Paths
        paths = [item['path'] for item in batch]

        return {
            'paths': paths,
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def find_images(folder_path):
    """查找指定文件夹下的常见图片文件"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [str(p.resolve()) for p in Path(folder_path).rglob("*") if p.suffix.lower() in extensions]

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class JoyCaptionRun:
    model_cache = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "preset_prompt": (preset_prompts, {"default": "Write a descriptive caption for this image in a formal tone.-以正式的语气为这张图片写一个描述性的标题。"}),
                "merge_prompt": ("BOOLEAN", {"default": False}),
                "model": (
                    [
                        "llama-joycaption-alpha-two-hf-llava-nf4",
                        "llama-joycaption-alpha-two-hf-llava",
                    ],
                    {"default": "llama-joycaption-alpha-two-hf-llava-nf4"},
                ),
                "use_cache": ("BOOLEAN", {"default": True}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 10, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "save_img_prompt_to_folder": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "images_dir": ("STRING", {"default": "", "multiline": False}),
                "tagger": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        model,
        prompt,
        preset_prompt,
        merge_prompt,
        max_new_tokens,
        do_sample,
        use_cache,
        temperature,
        top_k,
        top_p,
        save_img_prompt_to_folder="",
        batch_size=1,
        images_dir=None,
        tagger="",
        image=None,
        seed=0,
        unload_model=False,
    ):
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        nf4 = False
        if model == "llama-joycaption-alpha-two-hf-llava-nf4":
            nf4 = True
        
        if preset_prompt != "None":
            if merge_prompt:
                prompt = f"{preset_prompt.rsplit('-', 1)[0]}. {prompt}"
            prompt = preset_prompt.rsplit("-", 1)[0]
        
        model = os.path.join(model_path, model)
        if JoyCaptionRun.model_cache is None:
            JoyCaptionRun.model_cache = JoyCaption(model, nf4)
        
        JC = JoyCaptionRun.model_cache

        save_img_prompt_to_folder = save_img_prompt_to_folder.strip() if save_img_prompt_to_folder.strip() != "" else None

        if images_dir.strip() != "":
            JC.inference(
                prompt, 
                batch_size, 
                max_new_tokens, 
                do_sample, 
                use_cache, 
                temperature, 
                top_k, 
                top_p, 
                save_img_prompt_to_folder=save_img_prompt_to_folder, 
                images_dir=images_dir.strip(), 
                tagger=tagger,
                image=None,
                )
            caption = f"All prompt files are saved in the directory {images_dir.strip()} with the same file name as the image"
        elif image is not None:
            caption = JC.inference(
                prompt, 
                batch_size, 
                max_new_tokens, 
                do_sample, 
                use_cache, 
                temperature, 
                top_k, 
                top_p, 
                save_img_prompt_to_folder=save_img_prompt_to_folder, 
                images_dir=None, 
                tagger=tagger,
                image=image,
                )
        else:
            raise ValueError("Either 'image' or 'images_dir' must be provided")
        
        if unload_model:
            import gc
            JC.clean()
            JC = None
            JoyCaptionRun.model_cache = None
            gc.collect()
            torch.cuda.empty_cache()

        return (caption,)
    
