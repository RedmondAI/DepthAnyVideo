import argparse
import logging
import os
import random

from easydict import EasyDict
import numpy as np
import torch
from dav.pipelines import DAVPipeline
from dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from dav.utils import img_utils

# **Edit 2:** Set global default dtype to float32
torch.set_default_dtype(torch.float32)

def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cast_to_float32(module):
    """
    Recursively cast all submodules and their parameters to float32.
    """
    for child in module.children():
        cast_to_float32(child)
    if isinstance(module, torch.nn.Module):
        module.float()
        # **Edit 3:** Ensure buffers are also cast to float32
        for buffer_name, buffer in module.named_buffers():
            module.register_buffer(buffer_name, buffer.float())


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run video depth estimation using Depth Any Video."
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default="hhyangcs/depth-any-video",
        help="Checkpoint path or hub name.",
    )

    # data setting
    parser.add_argument(
        "--data_dir", type=str, required=True, help="input data directory or file."
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=3,
        help="Denoising steps, 1-3 steps work fine.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to infer per forward",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=16,
        help="Number of frames to decode per forward",
    )
    parser.add_argument(
        "--num_interp_frames",
        type=int,
        default=16,
        help="Number of frames for inpaint inference",
    )
    parser.add_argument(
        "--num_overlap_frames",
        type=int,
        default=6,
        help="Number of frames to overlap between windows",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=1024,  # decrease for faster inference and lower memory usage
        help="Maximum resolution for inference.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

    if cfg.seed is None:
        import time

        cfg.seed = int(time.time())
    seed_all(cfg.seed)

    device = torch.device("cuda")

    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.info(f"output dir = {cfg.output_dir}")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.model_base, subfolder="vae", torch_dtype=torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg.model_base, subfolder="scheduler", torch_dtype=torch.float32)
    unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(cfg.model_base, subfolder="unet", torch_dtype=torch.float32)
    unet_interp = UNetSpatioTemporalRopeConditionModel.from_pretrained(cfg.model_base, subfolder="unet_interp", torch_dtype=torch.float32)

    # Force all model components to use float32 recursively
    vae = vae.float()
    unet = unet.float()
    unet_interp = unet_interp.float()

    # **Edit 1:** Recursively cast all submodules within `unet` and `unet_interp` to float32
    cast_to_float32(unet)
    cast_to_float32(unet_interp)

    # **Edit 4:** Explicitly cast the `time_embedding` submodules to float32
    # This addresses potential nested modules that might retain float16
    if hasattr(unet, 'time_embedding'):
        cast_to_float32(unet.time_embedding)
    if hasattr(unet_interp, 'time_embedding'):
        cast_to_float32(unet_interp.time_embedding)

    pipe = DAVPipeline(
        vae=vae,
        unet=unet,
        unet_interp=unet_interp,
        scheduler=scheduler,
    )
    pipe.num_inference_steps = cfg.denoise_steps  # Set the number of inference steps
    pipe = pipe.to(device, dtype=torch.float32)  # Ensure pipeline is in float32

    file_name = cfg.data_dir.split("/")[-1].split(".")[0]
    is_video = cfg.data_dir.endswith(".mp4")

    if os.path.isdir(cfg.data_dir):
        # Handle directory of images
        image_files = [
            os.path.join(cfg.data_dir, f)
            for f in os.listdir(cfg.data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for img_path in image_files:
            logging.info(f"Processing image: {img_path}")
            image = img_utils.read_image(img_path)
            image = img_utils.imresize_max(image, cfg.max_resolution)
            image = img_utils.imcrop_multi(image)
            image_tensor = torch.stack([
                torch.from_numpy(_img / 255.0).permute(2, 0, 1).float() for _img in image  # Ensure tensor is float32
            ]).to(device)

            # Disable automatic mixed precision to ensure all operations use float32
            with torch.cuda.amp.autocast(enabled=False):
                # **Edit 5:** Add debug statements to verify tensor dtypes before inference
                print(f"image_tensor dtype: {image_tensor.dtype}")
                for name, param in unet.named_parameters():
                    print(f"Unet parameter '{name}' dtype: {param.dtype}")
                for name, param in unet.time_embedding.named_parameters():
                    print(f"Unet Time Embedding parameter '{name}' dtype: {param.dtype}")
                
                pipe_out = pipe(
                    image_tensor,
                    num_frames=cfg.num_frames,
                    num_overlap_frames=cfg.num_overlap_frames,
                    num_interp_frames=cfg.num_interp_frames,
                    decode_chunk_size=cfg.decode_chunk_size,
                )

            disparity = pipe_out.disparity
            disparity_colored = pipe_out.disparity_colored
            image = pipe_out.image
            # (N, H, 2 * W, 3)
            merged = np.concatenate(
                [
                    image,
                    disparity_colored,
                ],
                axis=2,
            )

            output_file = os.path.join(
                cfg.output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.png"
            )
            img_utils.write_image(output_file, merged[0])
            logging.info(f"Saved output to {output_file}")
    else:
        if is_video:
            num_interp_frames = cfg.num_interp_frames
            num_overlap_frames = cfg.num_overlap_frames
            num_frames = cfg.num_frames
            assert num_frames % 2 == 0, "num_frames should be even."
            assert (
                2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
            ), "Invalid frame overlap."
            max_frames = (num_interp_frames + 2 - num_overlap_frames) * (num_frames // 2)
            image, fps = img_utils.read_video(cfg.data_dir, max_frames=max_frames)
        else:
            image = img_utils.read_image(cfg.data_dir)

        image = img_utils.imresize_max(image, cfg.max_resolution)
        image = img_utils.imcrop_multi(image)
        image_tensor = torch.stack([
            torch.from_numpy(_img / 255.0).permute(2, 0, 1).float() for _img in image  # Ensure tensor is float32
        ]).to(device)

        # Disable automatic mixed precision to ensure all operations use float32
        with torch.cuda.amp.autocast(enabled=False):
            # **Edit 5:** Add debug statements to verify tensor dtypes before inference
            print(f"image_tensor dtype: {image_tensor.dtype}")
            for name, param in unet.named_parameters():
                print(f"Unet parameter '{name}' dtype: {param.dtype}")
            for name, param in unet.time_embedding.named_parameters():
                print(f"Unet Time Embedding parameter '{name}' dtype: {param.dtype}")
            
            pipe_out = pipe(
                image_tensor,
                num_frames=cfg.num_frames,
                num_overlap_frames=cfg.num_overlap_frames,
                num_interp_frames=cfg.num_interp_frames,
                decode_chunk_size=cfg.decode_chunk_size,
            )

        disparity = pipe_out.disparity
        disparity_colored = pipe_out.disparity_colored
        image = pipe_out.image
        # (N, H, 2 * W, 3)
        merged = np.concatenate(
            [
                image,
                disparity_colored,
            ],
            axis=2,
        )

        if is_video:
            img_utils.write_video(
                os.path.join(cfg.output_dir, f"{file_name}.mp4"),
                merged,
                fps,
            )
        else:
            img_utils.write_image(
                os.path.join(cfg.output_dir, f"{file_name}.png"),
                merged[0],
            )
