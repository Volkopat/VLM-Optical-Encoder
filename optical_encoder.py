"""
Optical Encoder: Merged DeepEncoder + Adapter

This is a SINGLE unified module that:
1. Loads DeepEncoder (frozen)
2. Auto-detects VLM vision dimension
3. Creates and trains adapter automatically

Usage:
    # Auto-detects dimension from VLM
    encoder = OpticalEncoder(vlm_model)

    # Train adapter only (DeepEncoder frozen)
    encoder.train_adapter(dataset, vlm_model)

    # Inference
    vision_tokens = encoder(images)
"""

import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np


class UniversalAdapter(nn.Module):
    """
    Universal adapter that bridges DeepEncoder to any VLM

    Architecture:
    - Input: DeepEncoder tokens [batch, num_pages*256, 1280]
    - Output: VLM vision tokens [batch, num_pages*256, target_dim]
    """

    def __init__(
        self,
        input_dim=1280,
        target_dim=2048,
        adapter_type='mlp',
        hidden_dim=3072,
        max_pages=200,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.adapter_type = adapter_type
        self.max_pages = max_pages

        # Projection layer
        if adapter_type == 'linear':
            self.projection = nn.Linear(input_dim, target_dim)
        elif adapter_type == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, target_dim),
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        # Layer norm
        self.layer_norm = nn.LayerNorm(target_dim)

        # Page position embeddings
        self.page_embeddings = nn.Embedding(max_pages, target_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small values for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, num_pages=None):
        """
        Args:
            x: DeepEncoder tokens [batch, num_pages*256, input_dim]
            num_pages: Number of pages

        Returns:
            Vision tokens [batch, num_pages*256, target_dim]
        """
        batch_size, seq_len, dim = x.shape

        # Auto-detect number of pages
        if num_pages is None:
            num_pages = seq_len // 256
            if seq_len % 256 != 0:
                num_pages = 1

        # Project to target dimension
        x = self.projection(x)

        # Add page position embeddings for multi-page
        if num_pages > 1:
            page_indices = torch.arange(num_pages, device=x.device)
            page_indices = page_indices.repeat_interleave(256)
            page_embeds = self.page_embeddings(page_indices)
            page_embeds = page_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            x = x + page_embeds

        # Normalize
        x = self.layer_norm(x)

        return x

    def get_num_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OpticalEncoder(nn.Module):
    """
    Unified Optical Encoder = DeepEncoder + Adapter

    Automatically detects VLM dimension and creates appropriate adapter
    """

    def __init__(
        self,
        vlm_model=None,
        target_dim=None,
        deepencoder_path=None,
        adapter_checkpoint=None,
        adapter_type='mlp',
        max_pages=200,
        device='cuda'
    ):
        """
        Args:
            vlm_model: VLM model to auto-detect dimension (optional if target_dim provided)
            target_dim: Manual override for target dimension
            deepencoder_path: Path to DeepEncoder weights (default: ./models/DeepEncoder)
            adapter_checkpoint: Path to pretrained adapter (optional)
            adapter_type: Adapter type ('mlp' or 'linear')
            max_pages: Maximum pages supported
            device: Device to run on
        """
        super().__init__()

        self.device = device
        self.max_pages = max_pages
        self.adapter_type = adapter_type

        # 1. Load DeepEncoder (FROZEN)
        self._load_deepencoder(deepencoder_path)

        # 2. Detect or use target dimension
        if target_dim is None:
            if vlm_model is None:
                raise ValueError("Must provide either vlm_model or target_dim!")
            target_dim = self._detect_vlm_dimension(vlm_model)

        self.target_dim = target_dim
        print(f"✓ Detected VLM vision dimension: {target_dim}")

        # 3. Create Adapter (TRAINABLE)
        self._create_adapter(target_dim)

        # 4. Load pretrained adapter if provided
        if adapter_checkpoint:
            self.load_adapter(adapter_checkpoint)

    def _load_deepencoder(self, deepencoder_path):
        """Load DeepEncoder components (SAM + CLIP + Projector)"""
        from deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
        from easydict import EasyDict as adict

        if deepencoder_path is None:
            # Default: Download from HuggingFace
            deepencoder_path = "Volkopat/DeepSeek-DeepEncoder"

        # Check if it's a HuggingFace repo ID (contains "/" and not a local path)
        if "/" in deepencoder_path and not os.path.exists(deepencoder_path):
            print(f"Downloading DeepEncoder from HuggingFace: {deepencoder_path}...")
            from huggingface_hub import snapshot_download

            # Download to cache
            cache_dir = snapshot_download(
                repo_id=deepencoder_path,
                repo_type="model",
                resume_download=True,
            )
            deepencoder_path = cache_dir
            print(f"✓ Downloaded to cache: {cache_dir}")

        print(f"Loading DeepEncoder from {deepencoder_path}...")

        # Load SAM
        self.sam = build_sam_vit_b(checkpoint=None)
        self.sam.load_state_dict(torch.load(os.path.join(deepencoder_path, "sam_encoder.pth")))
        self.sam = self.sam.to(self.device).half()
        self.sam.eval()

        # Load CLIP
        self.clip = build_clip_l()
        self.clip.load_state_dict(torch.load(os.path.join(deepencoder_path, "clip_encoder.pth")))
        self.clip = self.clip.to(self.device).half()
        self.clip.eval()

        # Load Projector
        projector_cfg = adict({'projector_type': 'linear', 'input_dim': 2048, 'n_embed': 1280})
        self.projector = MlpProjector(projector_cfg)
        self.projector.load_state_dict(torch.load(os.path.join(deepencoder_path, "projector.pth")))
        self.projector = self.projector.to(self.device).half()
        self.projector.eval()

        # Freeze DeepEncoder
        for param in self.sam.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = False

        print(f"✓ DeepEncoder loaded and frozen (401M params)")

    def _detect_vlm_dimension(self, vlm_model):
        """
        Auto-detect VLM's vision token dimension

        Works for:
        - Qwen3-VL: Check language_model.embed_tokens
        - LLaVA: Check language_model.model.embed_tokens
        - Most VLMs: Try common paths
        """
        # Try common paths
        possible_paths = [
            lambda m: m.language_model.embed_tokens.weight.shape[1],  # Qwen3-VL
            lambda m: m.model.embed_tokens.weight.shape[1],           # Some VLMs
            lambda m: m.language_model.model.embed_tokens.weight.shape[1],  # LLaVA
            lambda m: m.get_input_embeddings().weight.shape[1],       # Generic
        ]

        for path_fn in possible_paths:
            try:
                dim = path_fn(vlm_model)
                return dim
            except:
                continue

        raise ValueError(
            "Could not auto-detect VLM dimension! "
            "Please provide target_dim manually."
        )

    def _create_adapter(self, target_dim):
        """Create adapter from DeepEncoder (1280) to VLM (target_dim)"""
        self.adapter = UniversalAdapter(
            input_dim=1280,
            target_dim=target_dim,
            adapter_type=self.adapter_type,
            max_pages=self.max_pages
        )
        self.adapter = self.adapter.to(self.device).float()

        print(f"✓ Adapter created (1280 → {target_dim}, {self.adapter.get_num_params():,} params)")

    @torch.no_grad()
    def encode_images(self, images):
        """
        Encode images with DeepEncoder

        Args:
            images: List of PIL Images

        Returns:
            tokens_1280: [1, num_pages*256, 1280]
        """
        all_tokens = []

        for image in images:
            # Prepare image
            img_array = np.array(image.resize((1024, 1024)))
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device).half()

            # DeepEncoder forward
            sam_features = self.sam(img_tensor)
            clip_output = self.clip(img_tensor, sam_features)
            clip_no_cls = clip_output[:, 1:]
            sam_flat = sam_features.flatten(2).permute(0, 2, 1)
            combined = torch.cat((clip_no_cls, sam_flat), dim=-1)
            vision_tokens = self.projector(combined)  # [1, 256, 1280]

            all_tokens.append(vision_tokens)

        # Concatenate all pages
        return torch.cat(all_tokens, dim=1)  # [1, num_pages*256, 1280]

    def forward(self, images):
        """
        Full pipeline: Images → VLM vision tokens

        Args:
            images: List of PIL Images (1-200 pages)

        Returns:
            vision_tokens: [1, num_pages*256, target_dim]
        """
        # Encode with DeepEncoder (frozen)
        tokens_1280 = self.encode_images(images)

        # Adapt to VLM dimension (trainable)
        num_pages = len(images)
        tokens_vlm = self.adapter(tokens_1280.float(), num_pages=num_pages)

        return tokens_vlm.half()

    def freeze_deepencoder(self):
        """Freeze DeepEncoder (already frozen by default)"""
        for param in self.sam.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = False

    def unfreeze_adapter(self):
        """Unfreeze adapter for training"""
        for param in self.adapter.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Get trainable parameters (adapter only)"""
        return [p for p in self.adapter.parameters() if p.requires_grad]

    def load_adapter(self, checkpoint_path):
        """Load pretrained adapter"""
        self.adapter.load_state_dict(torch.load(checkpoint_path))
        print(f"✓ Loaded adapter from {checkpoint_path}")

    def save_adapter(self, save_path):
        """Save adapter"""
        torch.save(self.adapter.state_dict(), save_path)
        print(f"✓ Saved adapter to {save_path}")

    @classmethod
    def from_pretrained(cls, vlm_model, adapter_checkpoint, **kwargs):
        """
        Load with pretrained adapter

        Usage:
            encoder = OpticalEncoder.from_pretrained(
                vlm_model=qwen3_model,
                adapter_checkpoint="adapters/qwen3_vl_2b.pth"
            )
        """
        return cls(
            vlm_model=vlm_model,
            adapter_checkpoint=adapter_checkpoint,
            **kwargs
        )


# ============================================================================
# Helper Functions
# ============================================================================

def render_text_to_pages(text, font_size=10, img_size=1024, max_pages=200):
    """Render text to multiple page images"""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except:
        font = ImageFont.load_default()

    chars_per_line = img_size // (font_size // 2)
    line_height = font_size + 2
    lines_per_page = (img_size - 20) // line_height

    # Split into lines
    all_lines = [text[i:i+chars_per_line] for i in range(0, len(text), chars_per_line)]

    # Split into pages
    images = []
    for page_idx in range(0, len(all_lines), lines_per_page):
        if len(images) >= max_pages:
            break

        page_lines = all_lines[page_idx:page_idx + lines_per_page]
        img = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)

        y = 10
        for line in page_lines:
            draw.text((10, y), line, fill='black', font=font)
            y += line_height

        images.append(img)

    return images


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("OPTICAL ENCODER - MERGED DEEPENCODER + ADAPTER")
    print("="*80)

    # Example 1: Create encoder with auto-detection
    print("\n1. Auto-detect VLM dimension:")
    print("   encoder = OpticalEncoder(vlm_model=your_qwen3_model)")
    print("   # Automatically detects dimension and creates adapter")

    # Example 2: Manual dimension
    print("\n2. Manual dimension specification:")
    print("   encoder = OpticalEncoder(target_dim=2048)")
    print("   # For Qwen3-VL or other 2048-dim VLMs")

    # Example 3: Load pretrained
    print("\n3. Load with pretrained adapter:")
    print("   encoder = OpticalEncoder.from_pretrained(")
    print("       vlm_model=qwen3_model,")
    print("       adapter_checkpoint='adapters/qwen3_vl_2b.pth'")
    print("   )")

    # Example 4: Training
    print("\n4. Training:")
    print("   encoder = OpticalEncoder(vlm_model=qwen3_model)")
    print("   encoder.freeze_deepencoder()  # Already frozen")
    print("   encoder.unfreeze_adapter()    # Enable adapter training")
    print("   optimizer = AdamW(encoder.get_trainable_params())")
    print("   # Train loop...")

    print("\n" + "="*80)
    print("✓ Merged encoder ready!")
    print("="*80)
