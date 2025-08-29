from monai.data import WSIReader
import torch.nn as nn
from monai.networks.blocks import TransformerBlock
import torchvision.transforms as transforms
import requests
from io import BytesIO
from configs import *
from data_generator import *


# --- Define the Vision Transformer Model Pipeline ---
class WSI_EXAONEPath_Classifier(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()

        print("Initializing WSI EXAONEPath Classifier...")

        # Load frozen EXAONEPath
        self.backbone = self._load_frozen_exaonepath()

        # Aggregation layers (trainable)
        self.aggregator = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification head (trainable)
        self.classifier = nn.Linear(256, num_classes)

        # Print parameter counts
        self._print_parameter_info()

    def _load_frozen_exaonepath(self):
        """Load EXAONEPath from HuggingFace using official method with correct paths"""
        try:
            # Add the cloned repo to Python path
            import sys
            sys.path.append('/content/EXAONEPath')

            # Import from the cloned repository
            from EXAONEPath.vision_transformer import VisionTransformer

            # Login to HuggingFace
            login(token=hf_token)

            # Load using the huggingface
            model = VisionTransformer.from_pretrained(
                "LGAI-EXAONE/EXAONEPath",  
                use_auth_token=hf_token
            )

            print("Successfully loaded EXAONEPath from HuggingFace")

        except Exception as e:
            print(f"HuggingFace loading failed: {e}")
            print("Falling back to manual checkpoint loading...")

            # Fallback: Load manually 
            try:
                # Import from cloned repo
                import sys
                sys.path.append('/content/EXAONEPath')
                from EXAONEPath.vision_transformer import vit_base

                # Download checkpoint if not exists
                checkpoint_url = "https://github.com/LG-AI-EXAONE/EXAONEPath/releases/download/1.0.0/EXAONEPath.ckpt"
                checkpoint_path = "/content/EXAONEPath.ckpt"

                if not os.path.exists(checkpoint_path):
                    print("Downloading EXAONEPath checkpoint...")
                    import urllib.request
                    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
                    print(f"Downloaded checkpoint to {checkpoint_path}")
                
                #Load manually
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                state_dict = checkpoint['state_dict']
                model = vit_base(patch_size=16, num_classes=0)  # num_classes=0 for feature extraction
                msg = model.load_state_dict(state_dict, strict=False)
                print(f'Pretrained weights loaded with msg: {msg}')

            except Exception as e2:
                print(f"❌ Manual loading also failed: {e2}")
                # Last resort: random weights with warning
                import sys
                sys.path.append('/content/EXAONEPath')
                from EXAONEPath.vision_transformer import vit_base
                model = vit_base(patch_size=16, num_classes=0)
                print(" [WARNING] : Using random weights - performance will be poor!")

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        return model

    def _print_parameter_info(self):
        """Print information about trainable vs frozen parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        print(f"Model initialized:")
        print(f"   - EXAONEPath: FROZEN ({frozen_params:,} params)")
        print(f"   - Aggregator + Classifier: TRAINABLE ({trainable_params:,} params)")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")

    def forward(self, patches):
        # patches shape: [batch_size, num_patches, channels, height, width]
        batch_size, num_patches = patches.shape[:2]

        # Reshape for backbone processing
        patches_flat = patches.view(-1, *patches.shape[2:])  # [batch_size*num_patches, C, H, W]

        # Extract features using frozen EXAONEPath
        with torch.no_grad():
            features = self.backbone(patches_flat)  # [batch_size*num_patches, 768]

        # Reshape back to [batch_size, num_patches, 768]
        features = features.view(batch_size, num_patches, -1)

        # Aggregate features (mean pooling)
        aggregated_features = torch.mean(features, dim=1)  # [batch_size, 768]

        # Pass through trainable layers
        aggregated_features = self.aggregator(aggregated_features)  # [batch_size, 256]
        logits = self.classifier(aggregated_features)  # [batch_size, num_classes]

        return logits

    def get_feature_maps(self, patches):
        """For analysis - get intermediate features"""
        patch_features = self.extract_patch_features(patches)
        aggregated_features = torch.mean(patch_features, dim=1)
        processed_features = self.aggregator(aggregated_features)

        return {
            'patch_features': patch_features,
            'aggregated_features': aggregated_features,
            'processed_features': processed_features
        }


'''
MODEL SPECIFIC SETUP FUNCTIONS
'''

# Fix the model creation function
def create_wsi_model(num_classes):
    """Create the WSI classification model"""
    model = WSI_EXAONEPath_Classifier(num_classes=num_classes)
    return model

# Optimizer setup (only train aggregator + classifier)
def setup_optimizer(model, learning_rate=1e-4):
    """Setup optimizer for trainable parameters only"""

    # Get only trainable parameters (aggregator + classifier)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    # Lower learning rate for stability
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Simple scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,  # 50 epochs
        eta_min=1e-6
    )

    return optimizer, scheduler


def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item[0] is not None and item[1] is not None]

    if len(batch) == 0:
        return None, None

    # Use the default collate function for valid items
    return torch.utils.data.dataloader.default_collate(batch)