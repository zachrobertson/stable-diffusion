import os
import yaml
import torch
import random
import logging
import numpy as np

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def load_diffusion_model(
        ckpt: str="models/ldm/stable-diffusion-v1/model.ckpt",
        config: str = "server/configs/v1-inference.yaml",
        seed: int = 42,
        plms: bool = False,
        workers: bool = False
    ):
    logging.info(f"Setting seed globally, seed: {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = "1" if workers else "0"

    logging.info(f"Setting config from config file, config file: {config}")
    # Calls OmegaConf.load(opt.config)
    # Lets try to ignore the Omegaconf call for now since we are going to
    # simplify the script by removing some generalizations.
    # Instead lets just get a dictionary representation of the config
    # using pyyaml
    with open(config, "r") as stream:
        config = yaml.safe_load(stream)

    # load_model_from_config(config, opt.ckpt)
    # This is a custom method that uses the OmegaConf config
    # that was just created as well as the model ckpt file path
    # to load the model into memory
    logging.info(f"Loading model from ckpt file, file path: {ckpt}")
    state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]

    # Instantiate model architecture from config and load weights
    model = instantiate_from_config(config["model"])
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    return model, sampler

def txt2img(
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_iter: int = 2,
        n_samples: int = 3,
        scale: float = 7.5,
        downsampling_factor: int = 8,
        latent_channels: int = 4,
        ddim_steps: int = 50
    ):
    """BitSurf Stable Diffusion txt2img implementation. This method is designed
    to construct images given a variety of user method arguments including
    a text prompt to guide image generation.


    Args:
        seed (int, optional): Starting seed for image generation, 
            this can be set to a default value for replication testing. Defaults to 42.
    """

    model, sampler = load_diffusion_model()

    data = [n_samples * [prompt]]
    with torch.no_grad():
        with torch.autocast("cuda"):
            with model.ema_scope():
                samples = []
                for n in range(n_iter):
                    for prompts in data:
                        uc = model.get_learned_conditioning(n_samples * [""])
                        c = model.get_learned_conditioning(prompts)
                        shape = [latent_channels, height // downsampling_factor, width // downsampling_factor]
                        sampler_output, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=0.0,
                            x_T=None,
                        )

                        x_samples = model.decode_first_stage(sampler_output)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                        images = torch.from_numpy(x_samples).permute(0, 3, 1, 2)
                        samples.append(images)
    return samples


if __name__ == "__main__":
    txt2img(
        "A blue grass field filled with hot air balloons",
        width=256,
        height=256,
    )