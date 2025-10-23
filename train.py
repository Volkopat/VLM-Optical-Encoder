"""
Universal Training Script for Optical Compression

Train adapter for ANY VLM using the merged OpticalEncoder.
Auto-detects VLM dimension and trains adapter only (DeepEncoder frozen).

Usage:
    # For Qwen3-VL-2B (auto-detects 2048 dims)
    python train.py --vlm_model_path /path/to/Qwen3-VL-2B-Instruct

    # For LLaVA-1.5 (auto-detects 4096 dims)
    python train.py --vlm_model_path /path/to/llava-1.5-7b-hf

    # Manual dimension override
    python train.py --target_dim 2048 --vlm_type qwen3

    # Custom training config
    python train.py --vlm_model_path /path/to/model --max_pages 60 --num_epochs 20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Add DeepEncoder path
script_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(script_dir)
deepseek_dir = os.path.join(examples_dir, "initial", "models", "DeepSeek-OCR")
sys.path.insert(0, deepseek_dir)
sys.path.insert(0, script_dir)

from optical_encoder import OpticalEncoder, render_text_to_pages


class OpticalCompressionDataset(Dataset):
    """Dataset for training optical compression adapter using REAL text"""

    def __init__(
        self,
        dataset_name='wikipedia',
        dataset_config='20220301.en',
        num_samples=1000,
        chars_per_doc=(5000, 100000),
        max_pages=6,
        seed=42,
        split='train'
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name ('wikipedia', 'c4', 'bookcorpus', etc.)
            dataset_config: Dataset configuration (e.g., '20220301.en' for Wikipedia)
            num_samples: Number of documents to use
            chars_per_doc: Range of characters per document (min, max)
            max_pages: Maximum pages to render per document
            seed: Random seed for reproducibility
            split: Dataset split to use
        """
        from datasets import load_dataset

        self.chars_per_doc = chars_per_doc
        self.max_pages = max_pages

        torch.manual_seed(seed)

        # Load real text dataset
        print(f"Loading {dataset_name} dataset...")
        try:
            if dataset_name == 'wikipedia':
                # Wikipedia: Use 'text' field
                raw_dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
                text_field = 'text'
            elif dataset_name == 'c4':
                # C4: Use 'text' field
                raw_dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True)
                text_field = 'text'
            elif dataset_name == 'bookcorpus':
                # BookCorpus: Use 'text' field
                raw_dataset = load_dataset('bookcorpus', split=split, streaming=True)
                text_field = 'text'
            else:
                # Generic dataset - try 'text' field
                raw_dataset = load_dataset(dataset_name, split=split, streaming=True)
                text_field = 'text'

            print(f"✓ Loaded {dataset_name}")

            # Filter and collect documents
            print(f"Collecting {num_samples} documents with {chars_per_doc[0]}-{chars_per_doc[1]} chars...")
            self.documents = []

            for sample in tqdm(raw_dataset, desc="Loading documents", total=num_samples):
                if len(self.documents) >= num_samples:
                    break

                text = sample[text_field]

                # Filter by length
                if chars_per_doc[0] <= len(text) <= chars_per_doc[1]:
                    self.documents.append(text)

            # If not enough documents in range, truncate/pad existing ones
            if len(self.documents) < num_samples:
                print(f"⚠️  Only found {len(self.documents)} documents in range, using what we have")
                # Collect more without length filter
                for sample in raw_dataset:
                    if len(self.documents) >= num_samples:
                        break

                    text = sample[text_field]

                    # Truncate or skip too short
                    if len(text) < chars_per_doc[0]:
                        continue
                    if len(text) > chars_per_doc[1]:
                        text = text[:chars_per_doc[1]]

                    self.documents.append(text)

            print(f"✓ Collected {len(self.documents)} real text documents")

        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            print(f"⚠️  Falling back to synthetic data (not recommended)")
            self._use_synthetic_fallback(num_samples, chars_per_doc)

    def _use_synthetic_fallback(self, num_samples, chars_per_doc):
        """Fallback to synthetic data if real dataset fails"""
        print(f"Generating {num_samples} synthetic documents...")
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "data", "analysis", "machine", "learning", "artificial", "intelligence",
        ]

        self.documents = []
        for i in range(num_samples):
            num_chars = torch.randint(chars_per_doc[0], chars_per_doc[1], (1,)).item()
            text = " ".join([words[torch.randint(0, len(words), (1,)).item()] for _ in range(num_chars // 5)])
            self.documents.append(text[:num_chars])

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        text = self.documents[idx]

        # Render to images
        images = render_text_to_pages(
            text,
            font_size=10,
            img_size=1024,
            max_pages=self.max_pages
        )

        return {
            'images': images,
            'text': text,
            'num_pages': len(images),
        }


def load_vlm_model(vlm_model_path, vlm_type='auto', quantization=None, device='cuda'):
    """
    Load VLM model (auto-detects type, supports quantization)

    Args:
        vlm_model_path: Path to VLM model
        vlm_type: Type of VLM ('qwen3', 'llava', 'auto')
        quantization: Quantization type ('int4', 'int8', or None)
        device: Device to load on

    Returns:
        vlm_model, processor
    """
    print(f"\nLoading VLM from: {vlm_model_path}")
    if quantization:
        print(f"  Quantization: {quantization.upper()}")

    if vlm_type == 'auto':
        # Auto-detect from model path
        if 'qwen' in vlm_model_path.lower():
            vlm_type = 'qwen3'
        elif 'llava' in vlm_model_path.lower():
            vlm_type = 'llava'
        else:
            print("⚠️  Could not auto-detect VLM type, assuming Qwen3-VL")
            vlm_type = 'qwen3'

    if vlm_type == 'qwen3':
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        # Setup quantization config
        quantization_config = None
        if quantization == 'int4':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("  ✓ Using INT4 quantization (NF4)")
        elif quantization == 'int8':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("  ✓ Using INT8 quantization")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model_path,
            torch_dtype=torch.float16 if not quantization else None,
            quantization_config=quantization_config,
            device_map=device if not quantization else "auto",
            trust_remote_code=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(
            vlm_model_path,
            trust_remote_code=True,
        )

        print(f"✓ Loaded Qwen3-VL model")

    elif vlm_type == 'llava':
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        model = LlavaForConditionalGeneration.from_pretrained(
            vlm_model_path,
            torch_dtype=torch.float16,
            device_map=device,
        ).eval()

        processor = AutoProcessor.from_pretrained(vlm_model_path)

        print(f"✓ Loaded LLaVA model")

    else:
        raise ValueError(f"Unsupported VLM type: {vlm_type}")

    # Freeze VLM
    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def train_epoch(encoder, dataloader, optimizer, vlm_model, processor, device='cuda'):
    """Train for one epoch"""
    encoder.train()
    encoder.unfreeze_adapter()  # Only adapter trainable
    encoder.freeze_deepencoder()  # Ensure DeepEncoder frozen

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch['images'][0]  # Unbatch (batch_size=1)
        text = batch['text'][0]
        num_pages = batch['num_pages'][0].item()

        # Forward through optical encoder
        vision_tokens = encoder(images)  # [1, num_pages*256, target_dim]

        # Get VLM text embeddings
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)

        text_embeds = vlm_model.language_model.embed_tokens(inputs['input_ids'])

        # Compute alignment loss (MSE between avg vision and avg text)
        vision_avg = vision_tokens.mean(dim=1)  # [1, target_dim]
        text_avg = text_embeds.mean(dim=1)  # [1, target_dim]

        loss = nn.functional.mse_loss(vision_avg, text_avg.detach().half())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Free memory
        del vision_tokens, text_embeds, vision_avg, text_avg
        torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Universal Optical Compression Training")

    # VLM configuration
    parser.add_argument('--vlm_model_path', type=str, default=None,
                        help='Path to VLM model (for auto-detection)')
    parser.add_argument('--vlm_type', type=str, default='auto',
                        choices=['auto', 'qwen3', 'llava'],
                        help='VLM type (auto-detected if not specified)')
    parser.add_argument('--target_dim', type=int, default=None,
                        help='Manual override for target dimension (auto-detected if not specified)')
    parser.add_argument('--quantization', type=str, default=None,
                        choices=['int4', 'int8'],
                        help='Quantization type for large models (int4 recommended for 8B on 12GB GPU)')

    # Dataset configuration
    parser.add_argument('--dataset_name', type=str, default='wikipedia',
                        choices=['wikipedia', 'c4', 'bookcorpus', 'custom'],
                        help='Dataset to use (wikipedia=high quality, c4=diverse, bookcorpus=books)')
    parser.add_argument('--dataset_config', type=str, default='20220301.en',
                        help='Dataset configuration (e.g., 20220301.en for Wikipedia)')

    # Training configuration
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--min_chars', type=int, default=5000,
                        help='Minimum characters per document')
    parser.add_argument('--max_chars', type=int, default=100000,
                        help='Maximum characters per document')
    parser.add_argument('--max_pages', type=int, default=6,
                        help='Maximum pages per document')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (must be 1 for variable pages)')

    # DeepEncoder configuration
    parser.add_argument('--deepencoder_path', type=str, default=None,
                        help='Path to DeepEncoder weights (auto-detected if not specified)')
    parser.add_argument('--adapter_checkpoint', type=str, default=None,
                        help='Path to pretrained adapter to continue training')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')

    args = parser.parse_args()

    print("="*80)
    print("UNIVERSAL OPTICAL COMPRESSION TRAINING")
    print("="*80)

    # Check requirements
    if args.vlm_model_path is None and args.target_dim is None:
        print("\n❌ ERROR: Must provide either --vlm_model_path (for auto-detection) or --target_dim (manual)")
        print("\nExamples:")
        print("  # Auto-detect from Qwen3-VL")
        print("  python train.py --vlm_model_path /path/to/Qwen3-VL-2B-Instruct")
        print("\n  # Manual dimension")
        print("  python train.py --target_dim 2048 --vlm_type qwen3")
        return

    # Load VLM (if provided)
    vlm_model = None
    processor = None
    if args.vlm_model_path:
        vlm_model, processor = load_vlm_model(
            args.vlm_model_path,
            args.vlm_type,
            args.quantization,
            args.device
        )

    # Create OpticalEncoder
    print("\n" + "="*80)
    print("CREATING OPTICAL ENCODER")
    print("="*80)

    encoder = OpticalEncoder(
        vlm_model=vlm_model,
        target_dim=args.target_dim,
        deepencoder_path=args.deepencoder_path,
        adapter_checkpoint=args.adapter_checkpoint,
        max_pages=args.max_pages * 10,  # Support more pages than training max
        device=args.device
    )

    print(f"\n✓ OpticalEncoder created:")
    print(f"  - DeepEncoder: 401M params (frozen)")
    print(f"  - Adapter: {encoder.adapter.get_num_params():,} params (trainable)")
    print(f"  - Target dimension: {encoder.target_dim}")

    # Create dataset
    print("\n" + "="*80)
    print("CREATING DATASET")
    print("="*80)

    dataset = OpticalCompressionDataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        chars_per_doc=(args.min_chars, args.max_chars),
        max_pages=args.max_pages,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Must be 0 for PIL images
    )

    print(f"✓ Dataset created: {len(dataset)} samples")
    print(f"  - Document size: {args.min_chars:,}-{args.max_chars:,} chars")
    print(f"  - Max pages: {args.max_pages}")

    # Create optimizer
    optimizer = optim.AdamW(
        encoder.get_trainable_params(),
        lr=args.learning_rate
    )

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    training_log = {
        'config': vars(args),
        'target_dim': encoder.target_dim,
        'adapter_params': encoder.adapter.get_num_params(),
        'epochs': [],
    }

    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        avg_loss = train_epoch(
            encoder, dataloader, optimizer, vlm_model, processor, args.device
        )

        print(f"\n✓ Epoch {epoch + 1} complete - Avg Loss: {avg_loss:.4f}")

        training_log['epochs'].append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat(),
        })

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"adapter_epoch_{epoch + 1}.pth")
            encoder.save_adapter(checkpoint_path)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "adapter_best.pth")
            encoder.save_adapter(best_path)
            print(f"✓ New best model saved (loss: {best_loss:.4f})")

    # Save final
    final_path = os.path.join(args.output_dir, "adapter_final.pth")
    encoder.save_adapter(final_path)

    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"  - Best: adapter_best.pth (loss: {best_loss:.4f})")
    print(f"  - Final: adapter_final.pth")
    print(f"  - Log: training_log.json")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("\n1. Test your adapter:")
    print(f"   python test.py --vlm_model_path {args.vlm_model_path} \\")
    print(f"                  --adapter_checkpoint {best_path}")
    print("\n2. Use in your code:")
    print(f"   from optical_encoder import OpticalEncoder")
    print(f"   encoder = OpticalEncoder.from_pretrained(")
    print(f"       vlm_model=your_vlm,")
    print(f"       adapter_checkpoint='{best_path}'")
    print(f"   )")


if __name__ == "__main__":
    main()
