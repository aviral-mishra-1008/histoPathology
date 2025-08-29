import os
import json
import pandas as pd
import openslide
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import HfApi, login, hf_hub_download
from monai.data import WSIReader, Dataset
from monai.transforms import (
    LoadImaged,
    Compose,
    RandSpatialCropd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)
import torchvision.transforms as transforms
import gc
from dotenv import load_dotenv

load_dotenv()

try:
    huggingface_token = os.getenv("huggingface")
    hf_token = huggingface_token
    login(token=huggingface_token)
except:
    print("Hugging Face token not found. Proceeding with public access.")

metadata_repo_id = "HistAI/HISTAI-metadata"
wsi_repo_id = "HistAI/HISTAI-colorectal-b2"


