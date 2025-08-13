import gc
import logging
import math
import os
import random
import shutil
import subprocess
from functools import partial

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import librosa
from pathlib import Path
import imageio
import torchvision
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.distributed as dist

from wan.dist import set_multi_gpus_devices
from wan.distributed.fsdp import shard_model
from wan.models.cache_utils import get_teacache_coefficients
from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline

from wan.utils.discrete_sampler import DiscreteSampling
from wan.utils.fp8_optimization import replace_parameters_by_name, convert_weight_dtype_wrapper, \
    convert_model_weight_to_float8
from wan.utils.utils import get_image_to_video_latent, save_videos_grid

def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_path, fps=25, quality=10):
    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None, saved_frames_dir=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        idx = 0
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            frame_path = os.path.join(saved_frames_dir, f"frame_{idx}.png")
            idx = idx + 1
            imageio.imwrite(frame_path, frame)
            writer.append_data(frame)
        writer.close()

    save_path_tmp = os.path.join(save_path, "video_without_audio.mp4")
    saved_frames_dir = os.path.join(save_path, "animated_images")
    os.makedirs(saved_frames_dir, exist_ok=True)

    # video_audio = (gen_video_samples + 1) / 2  # C T H W
    video_audio = (gen_video_samples / 2 + 0.5).clamp(0, 1)
    video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
    video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
    save_video(video_audio, save_path_tmp, fps=fps, quality=quality, saved_frames_dir=saved_frames_dir)

    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = os.path.join(save_path, "cropped_audio.wav")
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_path,
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list

    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# Global, lazily accessible inference pipeline loaded on import
_PIPELINE = None


def _is_dist_enabled() -> bool:
    try:
        return int(os.getenv("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


def _get_local_rank() -> int:
    try:
        return int(os.getenv("LOCAL_RANK", "0"))
    except Exception:
        return 0


def _ensure_dist_initialized():
    if not _is_dist_enabled():
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    try:
        torch.cuda.set_device(_get_local_rank())
    except Exception:
        pass


def _is_main_process() -> bool:
    if not _is_dist_enabled():
        return True
    try:
        return dist.get_rank() == 0
    except Exception:
        try:
            return int(os.getenv("RANK", "0")) == 0
        except Exception:
            return True


def _initialize_pipeline():
    """Build and move the StableAvatar pipeline to CUDA at import time.

    Uses checkpoint directories under `checkpoints/` by default. You can override via env vars:
      - STABLEAVATAR_PRETRAINED_DIR
      - STABLEAVATAR_WAV2VEC_DIR
      - STABLEAVATAR_CONFIG_PATH
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    base_dir = Path(__file__).resolve().parent
    checkpoints_dir = base_dir / "checkpoints"
    wan_dir = os.getenv("STABLEAVATAR_WAN_DIR", str(checkpoints_dir / "Wan2.1-Fun-V1.1-1.3B-InP"))
    pretrained_dir = os.getenv("STABLEAVATAR_PRETRAINED_DIR", str(checkpoints_dir / "StableAvatar-1.3B"))
    wav2vec_dir = os.getenv("STABLEAVATAR_WAV2VEC_DIR", str(checkpoints_dir / "wav2vec2-base-960h"))
    config_path_env = os.getenv("STABLEAVATAR_CONFIG_PATH", str(Path(pretrained_dir) / "config.yaml"))

    # Initialize distributed if launched with torchrun, and choose device
    _ensure_dist_initialized()
    device = f"cuda:{_get_local_rank()}" if _is_dist_enabled() else "cuda"
    weight_dtype = torch.bfloat16
    sampler_name = "Flow"

    # Try to load config if present; otherwise fall back to sensible defaults
    config = None
    if os.path.exists(config_path_env):
        try:
            config = OmegaConf.load(config_path_env)
            logger.info(f"Loaded config from: {config_path_env}")
        except Exception as ex:
            logger.warning(f"Failed to load config at {config_path_env}: {ex}. Proceeding with defaults.")
            config = None

    # Resolve subpaths with config fallbacks
    def cfg_get(cfg, section, key, default):
        if cfg is None:
            return default
        try:
            return cfg[section].get(key, default)
        except Exception:
            return default

    tokenizer_subpath = cfg_get(config, 'text_encoder_kwargs', 'tokenizer_subpath', 'google/umt5-xxl')
    text_encoder_subpath = cfg_get(config, 'text_encoder_kwargs', 'text_encoder_subpath', 'google/umt5-xxl')
    vae_subpath = cfg_get(config, 'vae_kwargs', 'vae_subpath', 'Wan2.1_VAE.pth')
    image_encoder_subpath = cfg_get(config, 'image_encoder_kwargs', 'image_encoder_subpath', 'xlm-roberta-large')
    transformer_subpath = cfg_get(config, 'transformer_additional_kwargs', 'transformer_subpath', 'transformer3d-square.pt')

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(wan_dir, tokenizer_subpath))

    text_kwargs = {
        'low_cpu_mem_usage': True,
        'torch_dtype': weight_dtype,
    }
    if config is not None and 'text_encoder_kwargs' in config:
        try:
            text_kwargs['additional_kwargs'] = OmegaConf.to_container(config['text_encoder_kwargs'])
        except Exception:
            pass

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(wan_dir, text_encoder_subpath),
        **text_kwargs,
    ).eval()

    vae_kwargs = {}
    if config is not None and 'vae_kwargs' in config:
        try:
            vae_kwargs['additional_kwargs'] = OmegaConf.to_container(config['vae_kwargs'])
        except Exception:
            pass

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(wan_dir, vae_subpath),
        **vae_kwargs,
    )

    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_dir)

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(wan_dir, image_encoder_subpath)
    ).eval()

    transformer_kwargs = {
        'low_cpu_mem_usage': False,
        'torch_dtype': weight_dtype,
    }
    if config is not None and 'transformer_additional_kwargs' in config:
        try:
            transformer_kwargs['transformer_additional_kwargs'] = OmegaConf.to_container(config['transformer_additional_kwargs'])
        except Exception:
            pass

    transformer3d = WanTransformer3DFantasyModel.from_pretrained(
        os.path.join(pretrained_dir, transformer_subpath),
        **transformer_kwargs,
    )
    if _is_dist_enabled():
        try:
            transformer3d.enable_multi_gpus_inference()
        except Exception as ex:
            logger.warning(f"enable_multi_gpus_inference failed: {ex}")

    Choosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[sampler_name]
    if config is not None and 'scheduler_kwargs' in config:
        scheduler = Choosen_Scheduler(
            **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
    else:
        scheduler = Choosen_Scheduler()

    pipeline = WanI2VTalkingInferenceLongPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer3d,
        clip_image_encoder=clip_image_encoder,
        scheduler=scheduler,
        wav2vec_processor=wav2vec_processor,
        wav2vec=wav2vec,
    )

    pipeline.to(device=device)

    _PIPELINE = pipeline
    if _is_dist_enabled():
        try:
            rank = dist.get_rank()
            world = dist.get_world_size()
        except Exception:
            rank, world = 0, int(os.getenv("WORLD_SIZE", "1"))
        logger.info(f"StableAvatar pipeline initialized on CUDA device {device} [rank {rank}/{world}].")
    else:
        logger.info("StableAvatar pipeline initialized and moved to CUDA.")
    return _PIPELINE


# Initialize on import
try:
    _initialize_pipeline()
except Exception as _ex:
    logger.error(f"Failed to initialize pipeline at import time: {_ex}")


def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
            optimize=False,
            lossless=True
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid_png_and_mp4(videos: torch.Tensor, rescale=False, n_rows=6, save_frames_path=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)
        outputs.append(x)

    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in outputs]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(save_frames_path, f'frame_{i}.png')
        pil_frame.save(save_path)


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


def generate(
    image_path: str,
    audio_path: str,
    prompt: str,
    pretrained_wav2vec_path: str = None,
    guidance_scale: float = 6.0,
    input_perturbation: float = 0.0,
    pretrained_model_name_or_path: str = None,
    transformer_path: str = None,
    revision: str = None,
    variant: str = None,
    output_dir: str = None,
    width: int = 512,
    height: int = 512,
    offload_model: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    t5_fsdp: bool = False,
    t5_cpu: bool = False,
    fsdp_dit: bool = False,
    seed: int = None,
    motion_frame: int = 25,
    sample_steps: int = None,
    sample_shift: float = None,
    sample_text_guide_scale: float = 5.0,
    sample_audio_guide_scale: float = 4.0,
    overlap_window_length: int = 10,
    config_path: str = None,
    enable_teacache: bool = False,
    teacache_threshold: float = 0.10,
    num_skip_start_steps: int = 5,
    teacache_offload: bool = False,
    GPU_memory_mode: str = "model_full_load",
    clip_sample_n_frames: int = 81,
):
    # Use globally initialized pipeline
    pipeline = _initialize_pipeline()
    vae = pipeline.vae
    fps = 25
    device = f"cuda:{_get_local_rank()}" if _is_dist_enabled() else "cuda"

    # Optional TeaCache enabling using the preloaded model directory
    if enable_teacache:
        try:
            coefficients = get_teacache_coefficients(os.getenv("STABLEAVATAR_PRETRAINED_DIR", str(Path(__file__).resolve().parent / "checkpoints" / "StableAvatar-1.3B")))
        except Exception:
            coefficients = None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients,
                sample_steps,
                teacache_threshold,
                num_skip_start_steps=num_skip_start_steps,
                offload=teacache_offload,
            )

    # RNG
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = torch.Generator(device=device)

    with torch.no_grad():
        video_length = int((clip_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if clip_sample_n_frames != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(image_path, None, video_length=video_length, sample_size=[height, width])
        sr = 16000
        vocal_input, sample_rate = librosa.load(audio_path, sr=sr)

        if isinstance(prompt, str):
            prompt = [prompt]

        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=sample_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            text_guide_scale=sample_text_guide_scale,
            audio_guide_scale=sample_audio_guide_scale,
            vocal_input_values=vocal_input,
            motion_frame=motion_frame,
            fps=fps,
            sr=sr,
            cond_file_path=image_path,
            seed=seed,
            overlap_window_length=overlap_window_length,
        ).videos

        saved_frames_dir = os.path.join(output_dir, "animated_images")
        os.makedirs(saved_frames_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"video_without_audio.mp4")

        if _is_main_process():
            save_videos_grid(sample, video_path, fps=fps)
