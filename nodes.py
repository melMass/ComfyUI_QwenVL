import os

import folder_paths
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

try:
    from qwen_omni_utils import process_mm_info

    QWEN_OMNI_AVAILABLE = True
except ImportError:
    QWEN_OMNI_AVAILABLE = False
    print("qwen_omni_utils not available. Install with: pip install qwen-omni-utils")


def tensor_to_pil(image_tensor, batch_index=0) -> Image.Image:
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def save_video_to_temp(video_input):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name

    video_input.save_to(temp_path)
    return temp_path


def save_audio_comfy_to_temp(audio_data):
    """Save ComfyUI audio format to temporary file and return path."""
    import tempfile

    # ComfyUI audio format: {"waveform": tensor, "sample_rate": int}
    waveform = audio_data["waveform"]
    sample_rate = audio_data["sample_rate"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        if len(waveform_np.shape) == 3:
            waveform_np = waveform_np[0]

        if len(waveform_np.shape) == 2:
            audio_np = waveform_np[0] if waveform_np.shape[0] == 1 else waveform_np.T
        else:
            audio_np = waveform_np

        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

        sf.write(temp_file.name, audio_np, samplerate=sample_rate)
        temp_path = temp_file.name
    return temp_path


def convert_audio_to_comfy_format(audio_tensor, sample_rate=24000):
    """Convert audio tensor from model to ComfyUI AUDIO format."""
    if len(audio_tensor.shape) == 1:
        # add batch and channel dimensions
        waveform = audio_tensor.unsqueeze(0).unsqueeze(0)
    elif len(audio_tensor.shape) == 2:
        # add batch
        waveform = audio_tensor.unsqueeze(0)
    else:
        waveform = audio_tensor

    return {"waveform": waveform, "sample_rate": sample_rate}


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
    "OMNI": [
        "Qwen2.5-Omni-7B",
    ],
}


class QwenLoadLModel:
    """Node to load Qwen LLM/VLM/OMNI models with optional quantization and features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    MODELS["VLM"] + MODELS["LLM"] + MODELS["OMNI"],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
            },
            "optional": {
                "enable_audio_output": ("BOOLEAN", {"default": True}),
                "flash_attention": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("QWEN2_MODEL",)
    FUNCTION = "load"
    CATEGORY = "QwenVL"

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def bf16_support(self):
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    def load(
        self,
        *,
        model: str,
        quantization,
        enable_audio_output=True,
        flash_attention=False,
    ):
        is_vlm = model in MODELS["VLM"]
        is_omni = model in MODELS["OMNI"]

        if is_omni and not QWEN_OMNI_AVAILABLE:
            raise ImportError(
                """qwen_omni_utils not available.
                Install with: pip install qwen-omni-utils"""
            )

        if model.startswith("Qwen"):
            model_id = f"Qwen/{model}" if is_omni else f"qwen/{model}"
        else:
            model_id = f"Skywork/{model}"

        model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.bf16_support else torch.float16,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }

        if flash_attention and (is_vlm or is_omni):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if is_omni:
            processor = Qwen2_5OmniProcessor.from_pretrained(model_checkpoint)
            model_instance = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_checkpoint, **model_kwargs
            )
            # saves ~2GB memory
            if not enable_audio_output:
                model_instance.disable_talker()
            return (("OMNI", model_instance, processor, enable_audio_output),)

        elif is_vlm:
            processor = AutoProcessor.from_pretrained(model_checkpoint)
            model_instance = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_checkpoint, **model_kwargs
            )
            return (("VLM", model_instance, processor, None),)

        else:
            model_instance = AutoModelForCausalLM.from_pretrained(
                model_checkpoint, **model_kwargs
            )
            processor = AutoTokenizer.from_pretrained(model_checkpoint)
            return (("LLM", model_instance, processor, None),)


class Qwen2VL:
    """Node to run inference with Qwen VLM models.

    supporting text, image, and video inputs.
    """

    def __init__(self):
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
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "output_mode": (
                    (
                        "string",
                        "batch",
                    ),
                    {"default": "string"},
                ),
                "image": ("IMAGE",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "inference"
    CATEGORY = "QwenVL"

    def inference(
        self,
        *,
        user,
        system,
        model,
        # keep_model_loaded,
        temperature,
        max_new_tokens,
        batch_size,
        seed,
        force_offload=True,
        output_mode="string",
        image=None,
        video=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model_type, qmodel, preprocessor, _ = model

        if torch.cuda.is_available():
            qmodel = qmodel.cuda()

        is_vlm = model_type == "VLM"

        if not is_vlm and not user.strip():
            raise ValueError("LM models need a prompt")

        with torch.no_grad():
            pil_image = None
            processed_video_path = None

            user_content = [{"type": "text", "text": user}]
            text_list = []
            image_list = []

            if is_vlm:
                if video is not None:
                    print("Processing video input")
                    processed_video_path = save_video_to_temp(video)
                    user_content.insert(
                        0, {"type": "video", "video": processed_video_path}
                    )

                elif image is not None:
                    print("deal image")
                    pil_image = tensor_to_pil(image)
                    user_content.insert(0, {"type": "image", "image": pil_image})

            else:
                pass

            for _ in range(batch_size):
                if is_vlm:
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

            inputs = preprocessor(
                text=text_list,
                images=image_list if pil_image else None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

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
                ) from e

            finally:
                if processed_video_path and os.path.exists(processed_video_path):
                    os.remove(processed_video_path)

                if force_offload:
                    _, qmodel, _, _ = model
                    qmodel.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()


class QwenOmni:
    """Node to run inference with Qwen Omni models.

    Supporting text, image, video, and audio inputs and outputs.
    """

    def __init__(self):
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
                        "default": "You are Qwen, a virtual human capable of perceiving auditory and visual inputs, as well as generating text and speech.",
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
                "use_audio_in_video": ("BOOLEAN", {"default": True}),
                "return_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Make sure to enable audio output when loading the model if you want audio responses.",
                    },
                ),
                "speaker": (
                    ["Chelsie", "Ethan"],
                    {"default": "Chelsie"},
                ),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "output_mode": (
                    (
                        "string",
                        "batch",
                    ),
                    {"default": "string"},
                ),
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("text_output", "batch_output", "audio_output")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "inference"
    CATEGORY = "QwenVL"

    def inference(
        self,
        *,
        user,
        system,
        model,
        temperature,
        max_new_tokens,
        batch_size,
        seed,
        use_audio_in_video,
        return_audio,
        speaker,
        output_mode="batch",
        image=None,
        video=None,
        audio=None,
        force_offload=True,
    ):
        if not QWEN_OMNI_AVAILABLE:
            raise ImportError(
                "qwen_omni_utils not available. Install with: pip install qwen-omni-utils"
            )

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model_type, qmodel, preprocessor, audio_output_enabled = model

        if torch.cuda.is_available():
            qmodel = qmodel.cuda()

        if model_type != "OMNI":
            raise ValueError(
                "QwenOmni node requires an OMNI model. Please use the correct model type."
            )

        if return_audio and not audio_output_enabled:
            return_audio = False
            print(
                "Warning: Audio output requested but model was loaded without audio support"
            )

        with torch.no_grad():
            pil_image = None
            processed_video_path = None
            processed_audio_path = None

            conversation_content = [{"type": "text", "text": user}]

            if image is not None:
                pil_image = tensor_to_pil(image)
                conversation_content.insert(0, {"type": "image", "image": pil_image})

            if video is not None:
                processed_video_path = save_video_to_temp(video)
                conversation_content.insert(
                    0, {"type": "video", "video": processed_video_path}
                )

            if audio is not None:
                processed_audio_path = save_audio_comfy_to_temp(audio)
                conversation_content.insert(
                    0, {"type": "audio", "audio": processed_audio_path}
                )

            conversations = []
            for _ in range(batch_size):
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system}],
                    },
                    {
                        "role": "user",
                        "content": conversation_content.copy(),
                    },
                ]
                conversations.append(conversation)

            try:
                text_list = []
                results = []
                audio_output = None

                for conversation in conversations:
                    text = preprocessor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )
                    text_list.append(text)

                for i, (conversation, text) in enumerate(zip(conversations, text_list)):
                    mm_result = process_mm_info(
                        conversation, use_audio_in_video=use_audio_in_video
                    )
                    if isinstance(mm_result, tuple):
                        if len(mm_result) == 4:
                            audios, images, videos, _ = mm_result
                        elif len(mm_result) == 3:
                            audios, images, videos = mm_result
                        else:
                            raise ValueError(
                                f"Unexpected return format from process_mm_info: {len(mm_result)} elements"
                            )
                    else:
                        raise ValueError("process_mm_info should return a tuple")

                    inputs = preprocessor(
                        text=text,
                        audio=audios,
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=use_audio_in_video,
                    )
                    inputs = inputs.to(qmodel.device)

                    if return_audio and audio_output_enabled:
                        text_ids, audio = qmodel.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            use_audio_in_video=use_audio_in_video,
                            return_audio=True,
                            speaker=speaker,
                        )

                        if i == 0 and audio is not None:
                            audio_output = convert_audio_to_comfy_format(
                                audio, sample_rate=24000
                            )
                    else:
                        text_ids = qmodel.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            use_audio_in_video=use_audio_in_video,
                            return_audio=False,
                        )

                    text_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, text_ids)
                    ]

                    result_text = preprocessor.batch_decode(
                        text_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    results.extend(result_text)

                if output_mode == "string":
                    final_output = "\n\n---\n\n".join(results)
                else:
                    final_output = results

                return (
                    final_output,
                    results,
                    audio_output if audio_output else None,
                )

            except Exception as e:
                raise RuntimeError(f"Error during Qwen Omni inference: {str(e)}") from e

            finally:
                if processed_video_path and os.path.exists(processed_video_path):
                    os.remove(processed_video_path)
                if processed_audio_path and os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)

                if force_offload:
                    if qmodel is not None:
                        qmodel.cpu()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
