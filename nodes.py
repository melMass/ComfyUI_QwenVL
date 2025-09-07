import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from PIL import Image
import numpy as np
import folder_paths
import subprocess
import uuid
import comfy.model_management as mm


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


MODELS = {
    "VLM": [
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "SkyCaptioner-V1",
    ],
    "LLM": [
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
    ],
}


class QwenLoadLModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    MODELS["VLM"] + MODELS["LLM"],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
                # "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
            }
        }

    RETURN_TYPES = ("QWEN2_MODEL",)
    FUNCTION = "load"
    CATEGORY = "QwenVL"


    @property
    def device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def bf16_support(self):
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    def load(self, model: str, quantization):
        if model.startswith("Qwen"):
            model_id = f"qwen/{model}"
        else:
            model_id = f"Skywork/{model}"

        IS_VLM = model in MODELS["VLM"]

        model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        tokenizer = None
        if IS_VLM:
            processor = AutoProcessor.from_pretrained(
                model_checkpoint,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
            processor = AutoTokenizer.from_pretrained(model_checkpoint)

        # mm.current_loaded_models.append(mm.LoadedModel(model))
        # mm.current_loaded_models.append(mm.LoadedModel(processor))

        return ((IS_VLM, model, processor),)


class Qwen2VL:
    def __init__(self):
        # self.model_checkpoint = None
        # self.processor = None
        # self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN2_MODEL",),
                "system": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "user": ("STRING", {"default": "", "multiline": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "output_mode": (
                    (
                        "string",
                        "batch",
                    ),
                    {"default": "batch"},
                ),
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "inference"
    CATEGORY = "QwenVL"

    def inference(
        self,
        user,
        system,
        model,
        # keep_model_loaded,
        temperature,
        max_new_tokens,
        batch_size,
        seed,
        output_mode="batch",
        image=None,
        video_path=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        IS_VLM, qmodel, preprocessor = model
        if not IS_VLM and not user.strip():
            raise ValueError("LM models need a prompt")

        with torch.no_grad():
            pil_image = None
            processed_video_path = None

            user_content = [{"type": "text", "text": user}]
            text_list = []
            image_list = []

            if IS_VLM:
                if video_path:
                    print("deal video_path", video_path)
                    unique_id = uuid.uuid4().hex
                    processed_video_path = f"/tmp/processed_video_{unique_id}.mp4"
                    ffmpeg_command = [
                        "ffmpeg",
                        "-i",
                        video_path,
                        "-vf",
                        "fps=1,scale='min(256,iw)':min'(256,ih)':force_original_aspect_ratio=decrease",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        processed_video_path,
                    ]
                    subprocess.run(ffmpeg_command, check=True)

                    user_content.insert(
                        0, {"type": "video", "video": processed_video_path}
                    )

                elif image is not None:
                    print("deal image")
                    pil_image = tensor_to_pil(image)
                    user_content.insert(0, {"type": "image", "image": pil_image})

            else:
                pass

            for i in range(batch_size):
                if IS_VLM:
                    content = [{"type": "text", "text": user}]
                    if pil_image:
                        content.insert(0, {"type": "image", "image": pil_image})
                    elif processed_video_path:
                        content.insert(
                            0, {"type": "video", "video": processed_video_path}
                        )
                    message = [{"role": "user", "content": content}]
                else:
                    message = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]

                formatted_text = preprocessor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                text_list.append(formatted_text)

                image_list.append(pil_image)

            # 准备输入
            inputs = preprocessor(
                text=text_list,
                images=image_list if pil_image else None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 推理
            try:
                generated_ids = qmodel.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                result = preprocessor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                if output_mode == "string":
                    final_output = "\n\n---\n\n".join(result)

                else:
                    final_output = result

                return (final_output, result)

            except Exception as e:
                raise RuntimeError(
                    f"Error during model inference: {str(e)}",
                )

            finally:
                # if not keep_model_loaded:
                #     del preprocessor
                #     del qmodel
                #     torch.cuda.empty_cache()
                #     torch.cuda.ipc_collect()

                if video_path and processed_video_path:
                    os.remove(processed_video_path)
