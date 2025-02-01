import os
import argparse
import torch
import random
import logging
import numpy as np

import requests
import matplotlib.pyplot as plt
from torchvision import transforms

from PIL import Image, ImageDraw
from io import BytesIO

from torchvision.transforms import ToTensor, ToPILImage

import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"

colors = {'red': (250, 0, 45, 120), 'green': (100, 250, 0, 120), 'blue': (0, 40, 250, 120), 'gold':(255, 215, 0, 120), 'pink':(255,105,180, 120)}

def add_ddpm_noise(image, t=500):
    # Hyperparameters for DDPM (use your model's specific schedule)
    betas = np.linspace(0.0001, 0.02, 1000)  # Linear schedule as an example
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # Get alpha_t and sqrt(1 - alpha_t)
    alpha_t = alphas_cumprod[t]
    sqrt_alpha_t = np.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = np.sqrt(1 - alpha_t)

    # Load image
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (C, H, W) and normalize to [0, 1]
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Normalize to [-1, 1] (optional, depends on your model's normalization)
    img_tensor = (img_tensor - 0.5) * 2

    # Generate Gaussian noise
    noise = torch.randn_like(img_tensor)

    # Apply forward diffusion formula
    noisy_image = sqrt_alpha_t * img_tensor + sqrt_one_minus_alpha_t * noise

    # De-normalize back to [0, 1] for visualization (optional)
    noisy_image_vis = noisy_image / 2 + 0.5
    noisy_image_vis = torch.clamp(noisy_image_vis, 0, 1)

    # Convert back to PIL image for saving or displaying
    to_pil = transforms.ToPILImage()
    noisy_image_pil = to_pil(noisy_image_vis.squeeze(0))

    return noisy_image_pil

def generate_prompts(prompt, max_words=30):
    prompt_relevant_text = (
        # f"{prompt}"
        # f"Describe the image to answer: '{prompt}'. Respond in less than {max_words} words."
        # f"Describe the image to answer:'{prompt}'. Provide a response in less than {max_words} words."
        f"Describe all the information you extracted from the image that is relevant to answering the question: '{prompt}'. Provide a detailed summary within {max_words} words."
    )
    prompt_unrelevant_text = (
        f"Describe the image. Ignore the question '{prompt}' and focus only on visible details in less than {max_words} words."
    )
    # Describe the image less than {max_words} words.
    prompt_gt = f''
    return prompt_gt, prompt_relevant_text, prompt_unrelevant_text

def normalize_image(image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)
    return (image-mean)/std

def validate_prompt(prompt: str) -> str:
    if prompt.lower() == 'none':
        return ''
    return prompt

def image_parser(args):
    out = args.image_path.split(',')
    return out

def load_image(image_file):
    return Image.open(image_file).convert("RGB")

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def str_to_tuple(val):
    return tuple(map(int, val.strip('()').split(',')))

def log_all_args(args, logger):
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
def setup_logger(logger_name, log_file, level=logging.DEBUG):
    """Utility function to set up a logger with the given name and log file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Create console handler with the same log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def add_gaussian_noise(image, eps=2.0):
    """
    Add Gaussian noise to a PIL image with a maximum perturbation strength.

    Args:
        image (PIL.Image): Input image.
        eps (float): Maximum perturbation strength for each pixel.

    Returns:
        PIL.Image: Image with added Gaussian noise.
    """
    # Convert PIL image to tensor
    image_tensor = ToTensor()(image)

    # Calculate standard deviation for the noise
    std = 1

    noise_perturbation = torch.randn(image_tensor.size())
    # Generate Gaussian noise
    noise = torch.sign(noise_perturbation) * std

    noise = torch.clamp(noise, -eps/255, eps/255)
     
    # Add noise and clip the values to [0, 1]
    noisy_image_tensor = torch.clamp(image_tensor+noise, 0, 1)

    # Convert tensor back to PIL image
    noisy_image = ToPILImage()(noisy_image_tensor)

    return noisy_image