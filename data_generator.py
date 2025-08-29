import os
import torch
import numpy as np
import openslide
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import sys
from EXAONEPath.macenko import macenko_normalizer
from configs import *

def get_tissue_mask(slide_path):
    """Memory-safe tissue mask generation"""
    slide = None
    try:
        slide = openslide.OpenSlide(slide_path)
        level = slide.level_count - 1
        dimensions = slide.level_dimensions[level]
        thumbnail = slide.read_region((0, 0), level, dimensions).convert('L')
        thumbnail_np = np.array(thumbnail)
        _, tissue_mask = cv2.threshold(thumbnail_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Clear thumbnail from memory
        del thumbnail, thumbnail_np

        return tissue_mask

    except Exception as e:
        print(f"Error creating tissue mask: {e}")
        return None

    finally:
        # Always close slide, even on error
        if slide is not None:
            slide.close()
        # Force garbage collection
        gc.collect()

class WSI_MONAI_Dataset(Dataset):
    def __init__(self, repo_id, file_list, metadata, num_patches=32, patch_size=224, patch_level=0):
        """
        WSI dataset with proper dictionary metadata handling

        Args:
            repo_id: HuggingFace repository ID
            file_list: List of WSI file paths
            metadata: Dict with structure {case_id: {age, gender, icd10, ...}}
            num_patches: Number of patches per WSI
            patch_size: Size of patches (224 for EXAONEPath)
            patch_level: OpenSlide level for extraction
        """
        self.repo_id = repo_id
        self.file_list = file_list
        self.metadata = metadata
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_level = patch_level

        print("Creating consistent label mappings...")

        # Create label mappings
        self._create_label_mappings()

        # Initialize Macenko normalizer
        self.macenko_normalizer = macenko_normalizer()

        # ImageNet normalization
        self.imagenet_normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def _create_label_mappings(self):
        """Create label mappings from dictionary metadata"""

        valid_files = []
        valid_labels = []
        missing_cases = []

        print("Matching files with metadata...")

        for file_path in self.file_list:
            case_id = file_path.split('/')[0]

            # Check if case exists in metadata
            if case_id in self.metadata:
                case_info = self.metadata[case_id]

                # Get ICD-10 label
                if 'icd10' in case_info and case_info['icd10'] is not None:
                    icd10_code = str(case_info['icd10']).strip()

                    if icd10_code and icd10_code != 'nan' and icd10_code != '':
                        valid_files.append(file_path)
                        valid_labels.append(icd10_code)

                    else:
                        print(f"   ⚠️ {case_id}: Empty ICD-10 code")
                else:
                    print(f"   ⚠️ {case_id}: No 'icd10' key in metadata")
            else:
                missing_cases.append(case_id)

        # Report missing cases
        if missing_cases:
            print(f"   ❌ Cases not found in metadata: {missing_cases[:10]}...")  # Show first 10

        if len(valid_files) == 0:
            raise ValueError("❌ No files found with valid ICD-10 labels!")

        # Create final mappings
        self.file_list = valid_files
        self.icd10_labels = sorted(list(set(valid_labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.icd10_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Show label distribution
        from collections import Counter
        label_counts = Counter(valid_labels)

    def get_icd10_from_idx(self, idx):
        """Convert label index back to ICD-10 code"""
        return self.idx_to_label.get(idx, "Unknown")

    def get_case_info(self, case_id):
        """Get all info for a case"""
        return self.metadata.get(case_id, {})

    def _generate_tissue_mask(self, slide):
        """Generate tissue mask using Otsu thresholding"""
        level = slide.level_count - 1
        dimensions = slide.level_dimensions[level]
        thumbnail = slide.read_region((0, 0), level, dimensions).convert('L')
        thumbnail_np = np.array(thumbnail)
        _, tissue_mask = cv2.threshold(thumbnail_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return tissue_mask, slide.level_downsamples[level]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        case_id = file_path.split('/')[0]

        # Get ICD-10 label for this case
        if case_id in self.metadata:
            icd10_code = str(self.metadata[case_id]['icd10'])
            label = self.label_to_idx.get(icd10_code, -1)
        else:
            print(f"⚠️ No metadata found for case: {case_id}")
            return None, None

        if label == -1:
            print(f"⚠️ Unknown ICD-10 code '{icd10_code}' for case: {case_id}")
            return None, None

        # Download and process slide
        try:
            print(f"Downloading file: {file_path} (ICD-10: {icd10_code}) ......")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir="./wsi_cache"
            )

            slide = openslide.OpenSlide(local_path)

            # Generate tissue mask
            tissue_mask, downsample_factor = self._generate_tissue_mask(slide)
            tissue_coords = np.argwhere(tissue_mask > 0)

            if len(tissue_coords) == 0:
                print(f"⚠️ No tissue found in {file_path}")
                slide.close()
                if os.path.exists(local_path):
                    os.remove(local_path)
                return None, None

            # Extract patches from tissue regions
            patches_list = []
            max_attempts = self.num_patches * 3
            attempts = 0

            while len(patches_list) < self.num_patches and attempts < max_attempts:
                attempts += 1

                # Random tissue location
                y_m, x_m = tissue_coords[np.random.randint(len(tissue_coords))]
                x_level0 = int(x_m * downsample_factor)
                y_level0 = int(y_m * downsample_factor)

                try:
                    # Extract patch
                    patch = slide.read_region(
                        (x_level0, y_level0),
                        self.patch_level,
                        (self.patch_size, self.patch_size)
                    ).convert('RGB')

                    # Apply Macenko normalization
                    patch_macenko = self.macenko_normalizer(patch)

                    # Apply ImageNet normalization
                    patch_final = self.imagenet_normalize(patch_macenko)

                    patches_list.append(patch_final)

                except Exception as e:
                    continue

            slide.close()

            # Clean up downloaded file
            if os.path.exists(local_path):
                os.remove(local_path)

            if len(patches_list) == 0:
                print(f"❌ No valid patches extracted from {file_path}")
                return None, None

            # Pad with zeros if we don't have enough patches
            while len(patches_list) < self.num_patches:
                patches_list.append(torch.zeros_like(patches_list[0]))

            # Stack patches into tensor
            patches_tensor = torch.stack(patches_list)  # Shape: [num_patches, 3, 224, 224]

            return patches_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return None, None


# Keep the same custom collate function and denormalization function
def custom_collate_fn(batch):
    """Custom collate function to handle None values"""
    batch = [item for item in batch if item[0] is not None and item[1] is not None]

    if len(batch) == 0:
        return None, None

    patches = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])

    return patches, labels


def denormalize_imagenet(tensor):
    """Denormalize ImageNet-normalized tensor for display"""
    denorm = tensor.clone()
    denorm[0] = denorm[0] * 0.229 + 0.485
    denorm[1] = denorm[1] * 0.224 + 0.456
    denorm[2] = denorm[2] * 0.225 + 0.406
    return torch.clamp(denorm, 0, 1)