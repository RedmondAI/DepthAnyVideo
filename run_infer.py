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
    Recursively cast all submodules, parameters, and buffers to float32.
    """
    for child in module.children():
        cast_to_float32(child)
    if isinstance(module, torch.nn.Module):
        module.float()
        # **Edit 3:** Ensure buffers are also cast to float32
        for buffer_name, buffer in module.named_buffers():
            module.register_buffer(buffer_name, buffer.float())

        # **Edit 6:** Ensure that any internal tensors within the module are also float32
        for name, param in module.named_parameters(recurse=False):
            param.data = param.data.float()


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

    # **Edit 8:** Add forward hooks to enforce float32 on `time_embedding` module
    def enforce_float32_hook(module, input, output):
        return output.float()

    if hasattr(unet, 'time_embedding'):
        unet.time_embedding.register_forward_hook(enforce_float32_hook)
    if hasattr(unet_interp, 'time_embedding'):
        unet_interp.time_embedding.register_forward_hook(enforce_float32_hook)

    # **Edit 10:** Add forward pre-hooks to ensure `t_emb` is cast to float32 before entering `time_embedding`
    def enforce_float32_pre_hook(module, input):
        """
        Cast the input tensor to float32 to prevent dtype mismatches.
        """
        return (input[0].float(),)

    if hasattr(unet, 'time_embedding'):
        unet.time_embedding.register_forward_pre_hook(enforce_float32_pre_hook)
    if hasattr(unet_interp, 'time_embedding'):
        unet_interp.time_embedding.register_forward_pre_hook(enforce_float32_pre_hook)

    # **Edit 11:** Add forward pre-hooks for Conv2d layers to enforce float32 inputs
    def enforce_conv_float32_pre_hook(module, input):
        """
        Cast the input tensor to float32 before convolution operations to prevent dtype mismatches.
        """
        return (input[0].float(),)

    # Register the pre-hook for all Conv2d layers in `unet`
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_pre_hook(enforce_conv_float32_pre_hook)

    # Register the pre-hook for all Conv2d layers in `unet_interp`
    for name, module in unet_interp.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_pre_hook(enforce_conv_float32_pre_hook)

    # **Edit 12:** Add forward pre-hooks for Linear layers to enforce float32 inputs
    def enforce_linear_float32_pre_hook(module, input):
        """
        Cast the input tensor to float32 before linear operations to prevent dtype mismatches.
        """
        return (input[0].float(),)

    # Register the pre-hook for all Linear layers in `unet`
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(enforce_linear_float32_pre_hook)

    # Register the pre-hook for all Linear layers in `unet_interp`
    for name, module in unet_interp.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(enforce_linear_float32_pre_hook)

    # **Edit 14:** Add Forward Pre-Hooks for VAE Components to Enforce `float32` Inputs
    def enforce_float32_pre_hook(module, input):
        """
        Cast the input tensor to float32 before processing in VAE components to prevent dtype mismatches.
        """
        return (input[0].float(),)

    # Register the pre-hook for all Conv2d and Linear layers in `vae`
    for name, module in vae.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.register_forward_pre_hook(enforce_float32_pre_hook)

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

            # **Edit 9:** Add a forward hook to monitor the input dtype of `time_embedding`
            def check_dtype_hook(module, input, output):
                print(f"Input to {module.__class__.__name__}: dtype={input[0].dtype}")
                return output

            if hasattr(unet, 'time_embedding'):
                unet.time_embedding.register_forward_hook(check_dtype_hook)
            if hasattr(unet_interp, 'time_embedding'):
                unet_interp.time_embedding.register_forward_hook(check_dtype_hook)

            # Disable automatic mixed precision to ensure all operations use float32
            with torch.cuda.amp.autocast(enabled=False):
                # **Edit 5:** Add debug statements to verify tensor dtypes before inference
                print(f"image_tensor dtype: {image_tensor.dtype}")
                for name, param in unet.named_parameters():
                    print(f"Unet parameter '{name}' dtype: {param.dtype}")
                for name, param in unet.time_embedding.named_parameters():
                    print(f"Unet Time Embedding parameter '{name}' dtype: {param.dtype}")
                
                # **Edit 7:** Force all inputs to the pipeline to float32
                # This includes any additional tensors that might be created internally
                # If the pipeline or model creates tensors in float16, they need to be cast to float32
                # Since we can't modify the internal model code directly, we can use hooks or ensure inputs are float32

                # Example: If the pipeline accepts additional inputs that might default to float16, ensure they're float32
                # Assuming 'pipe' does not take additional hidden inputs, this should suffice

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
            
            # **Edit 7:** Force all inputs to the pipeline to float32
            # This includes any additional tensors that might be created internally
            # If the pipeline or model creates tensors in float16, they need to be cast to float32
            # Since we can't modify the internal model code directly, we can use hooks or ensure inputs are float32

            # Example: If the pipeline accepts additional inputs that might default to float16, ensure they're float32
            # Assuming 'pipe' does not take additional hidden inputs, this should suffice

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





