"""
Universal Testing Script for Optical Compression

Test trained adapter on ANY VLM with optical compression.
Supports custom texts, files, or LongBench v2 benchmark.

Usage:
    # Test on custom text
    python test.py --vlm_model_path /path/to/Qwen3-VL-2B-Instruct \\
                   --adapter_checkpoint adapters/qwen3_vl_2b.pth \\
                   --text "Your long document here..."

    # Test on file
    python test.py --vlm_model_path /path/to/model \\
                   --adapter_checkpoint adapters/adapter.pth \\
                   --file document.txt

    # Test on LongBench v2 (quick test)
    python test.py --vlm_model_path /path/to/model \\
                   --adapter_checkpoint adapters/adapter.pth \\
                   --benchmark longbench \\
                   --num_samples 20

    # Full LongBench v2 evaluation
    python test.py --vlm_model_path /path/to/model \\
                   --adapter_checkpoint adapters/adapter.pth \\
                   --benchmark longbench \\
                   --full
"""

import torch
import sys
import os
import argparse
import time
import json
import re
from tqdm import tqdm

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(script_dir)
deepseek_dir = os.path.join(examples_dir, "initial", "models", "DeepSeek-OCR")
sys.path.insert(0, deepseek_dir)
sys.path.insert(0, script_dir)

from optical_encoder import OpticalEncoder, render_text_to_pages


def load_vlm_model(vlm_model_path, vlm_type='auto', quantization=None, device='cuda'):
    """Load VLM model (auto-detects type)"""
    print(f"\nLoading VLM from: {vlm_model_path}")
    if quantization:
        print(f"Using {quantization.upper()} quantization")

    if vlm_type == 'auto':
        if 'qwen' in vlm_model_path.lower():
            vlm_type = 'qwen3'
        elif 'llava' in vlm_model_path.lower():
            vlm_type = 'llava'
        else:
            print("⚠️  Could not auto-detect VLM type, assuming Qwen3-VL")
            vlm_type = 'qwen3'

    if vlm_type == 'qwen3':
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        quantization_config = None
        if quantization == 'int4':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == 'int8':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

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

    return model, processor


@torch.no_grad()
def test_optical_compression(encoder, text, vlm_model, processor, question=None, device='cuda', max_pages=200):
    """
    Test with optical compression

    Args:
        encoder: OpticalEncoder instance
        text: Input text
        vlm_model: VLM model
        processor: VLM processor
        question: Optional question to ask (for Q&A)
        device: Device
        max_pages: Max pages to render

    Returns:
        dict with answer, tokens, time
    """
    # Render text to images
    images = render_text_to_pages(text, font_size=10, img_size=1024, max_pages=max_pages)
    num_pages = len(images)

    # Encode with optical encoder
    start = time.time()
    vision_tokens = encoder(images)  # [1, num_pages*256, target_dim]
    encode_time = time.time() - start

    # Prepare prompt
    if question:
        prompt = f"Based on the document shown in the image, answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = "Summarize the document shown in the image."

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)

    # Combine vision + text embeddings
    input_ids = inputs['input_ids']
    text_embeds = vlm_model.language_model.embed_tokens(input_ids)
    combined_embeds = torch.cat([vision_tokens, text_embeds], dim=1)

    vision_mask = torch.ones((1, vision_tokens.shape[1]), dtype=inputs['attention_mask'].dtype, device=device)
    combined_mask = torch.cat([vision_mask, inputs['attention_mask']], dim=1)

    # Generate
    start = time.time()
    output_ids = vlm_model.generate(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        max_new_tokens=512,
        do_sample=False,
    )
    gen_time = time.time() - start

    answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "assistant" in answer:
        answer = answer.split("assistant")[-1].strip()

    return {
        'answer': answer,
        'vision_tokens': vision_tokens.shape[1],
        'text_tokens': input_ids.shape[1],
        'total_tokens': vision_tokens.shape[1] + input_ids.shape[1],
        'num_pages': num_pages,
        'encode_time': encode_time,
        'gen_time': gen_time,
        'total_time': encode_time + gen_time,
    }


@torch.no_grad()
def test_native_text(text, vlm_model, processor, question=None, device='cuda', max_chars=500000):
    """
    Test with native text tokenization

    Args:
        text: Input text
        vlm_model: VLM model
        processor: VLM processor
        question: Optional question
        device: Device
        max_chars: Max characters (for truncation)

    Returns:
        dict with answer, tokens, time
    """
    # Truncate if needed
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    # Prepare prompt
    if question:
        prompt = f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"Context: {text}\n\nSummarize the context."

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)

    # Generate
    start = time.time()
    output_ids = vlm_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    gen_time = time.time() - start

    answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "assistant" in answer:
        answer = answer.split("assistant")[-1].strip()

    return {
        'answer': answer,
        'total_tokens': inputs['input_ids'].shape[1],
        'gen_time': gen_time,
        'truncated': truncated,
    }


def evaluate_accuracy(prediction, ground_truth):
    """Evaluate multiple choice accuracy (for LongBench)"""
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Exact match
    if pred_lower == gt_lower:
        return 1.0

    # Check if answer letter appears
    pattern = rf'\b{gt_lower}\b|\({gt_lower}\)|{gt_lower}\)'
    if re.search(pattern, pred_lower, re.IGNORECASE):
        return 1.0

    return 0.0


def test_longbench_v2(encoder, vlm_model, processor, device='cuda', num_samples=None, full=False):
    """
    Test on LongBench v2 benchmark

    Args:
        encoder: OpticalEncoder
        vlm_model: VLM model
        processor: VLM processor
        device: Device
        num_samples: Number of samples (None for full)
        full: Run full 503 samples

    Returns:
        dict with results
    """
    from datasets import load_dataset

    print("\n" + "="*80)
    print("LOADING LONGBENCH V2 DATASET")
    print("="*80)

    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    print(f"✓ Loaded {len(dataset)} samples")

    # Select samples
    if full:
        samples = list(dataset)
        num_samples = len(samples)
        print(f"✓ Running FULL benchmark: {num_samples} samples")
    elif num_samples:
        # Balanced sampling
        short_samples = [s for s in dataset if s.get('length') == 'short']
        medium_samples = [s for s in dataset if s.get('length') == 'medium']
        long_samples = [s for s in dataset if s.get('length') == 'long']

        samples_per_cat = num_samples // 3
        samples = (
            short_samples[:samples_per_cat] +
            medium_samples[:samples_per_cat] +
            long_samples[:num_samples - 2*samples_per_cat]
        )
        print(f"✓ Quick test: {len(samples)} balanced samples")
    else:
        samples = list(dataset)[:20]
        print(f"✓ Default: 20 samples")

    # Run benchmark
    print("\n" + "="*80)
    print("RUNNING BENCHMARK")
    print("="*80)

    results = {
        'optical': {'correct': 0, 'total': 0, 'failed': 0, 'total_tokens': 0, 'total_time': 0},
        'native': {'correct': 0, 'total': 0, 'failed': 0, 'total_tokens': 0, 'total_time': 0},
    }

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            context = sample['context']
            question = sample['question']
            choices = f"A. {sample['choice_A']}\nB. {sample['choice_B']}\nC. {sample['choice_C']}\nD. {sample['choice_D']}"
            full_question = f"{question}\n\n{choices}\n\nAnswer (A/B/C/D):"
            ground_truth = sample['answer']

            torch.cuda.empty_cache()

            # Test optical
            try:
                result_optical = test_optical_compression(
                    encoder, context, vlm_model, processor, full_question, device, max_pages=200
                )

                accuracy = evaluate_accuracy(result_optical['answer'], ground_truth)
                results['optical']['correct'] += accuracy
                results['optical']['total'] += 1
                results['optical']['total_tokens'] += result_optical['total_tokens']
                results['optical']['total_time'] += result_optical['total_time']

            except Exception as e:
                print(f"\n  Optical error on sample {idx}: {e}")
                results['optical']['failed'] += 1

            torch.cuda.empty_cache()

            # Test native
            try:
                result_native = test_native_text(
                    context, vlm_model, processor, full_question, device
                )

                accuracy = evaluate_accuracy(result_native['answer'], ground_truth)
                results['native']['correct'] += accuracy
                results['native']['total'] += 1
                results['native']['total_tokens'] += result_native['total_tokens']
                results['native']['total_time'] += result_native['gen_time']

            except Exception as e:
                # Native OOM/context exceeded is expected on long contexts
                results['native']['failed'] += 1

        except Exception as e:
            print(f"\n  Error on sample {idx}: {e}")
            continue

    return results


def print_results(results):
    """Print comparison results"""
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Calculate total samples for each method
    opt_total_samples = results['optical']['total'] + results['optical']['failed']
    nat_total_samples = results['native']['total'] + results['native']['failed']

    if results['optical']['total'] > 0:
        opt_acc = 100 * results['optical']['correct'] / results['optical']['total']
        opt_avg_tokens = results['optical']['total_tokens'] / results['optical']['total']
        opt_avg_time = results['optical']['total_time'] / results['optical']['total']
        opt_success_rate = 100 * results['optical']['total'] / opt_total_samples
        opt_overall_score = 100 * results['optical']['correct'] / opt_total_samples

        print(f"\nOptical Compression:")
        print(f"  Overall Score: {opt_overall_score:.1f}% ({results['optical']['correct']}/{opt_total_samples} samples)")
        print(f"  Success rate: {opt_success_rate:.1f}% ({results['optical']['total']}/{opt_total_samples} samples completed)")
        print(f"  Accuracy on completed: {opt_acc:.1f}% ({results['optical']['correct']:.1f}/{results['optical']['total']})")
        print(f"  Failed: {results['optical']['failed']} samples")
        print(f"  Avg tokens: {opt_avg_tokens:,.0f}")
        print(f"  Avg time: {opt_avg_time:.2f}s")

    if results['native']['total'] > 0:
        nat_acc = 100 * results['native']['correct'] / results['native']['total']
        nat_avg_tokens = results['native']['total_tokens'] / results['native']['total']
        nat_avg_time = results['native']['total_time'] / results['native']['total']
        nat_success_rate = 100 * results['native']['total'] / nat_total_samples
        nat_overall_score = 100 * results['native']['correct'] / nat_total_samples

        print(f"\nNative Text:")
        print(f"  Overall Score: {nat_overall_score:.1f}% ({results['native']['correct']}/{nat_total_samples} samples)")
        print(f"  Success rate: {nat_success_rate:.1f}% ({results['native']['total']}/{nat_total_samples} samples completed)")
        print(f"  Accuracy on completed: {nat_acc:.1f}% ({results['native']['correct']:.1f}/{results['native']['total']})")
        print(f"  Failed: {results['native']['failed']} samples (OOM/context exceeded)")
        print(f"  Avg tokens: {nat_avg_tokens:,.0f}")
        print(f"  Avg time: {nat_avg_time:.2f}s")

    if results['optical']['total'] > 0 and results['native']['total'] > 0:
        compression = nat_avg_tokens / opt_avg_tokens if opt_avg_tokens > 0 else 0
        savings = nat_avg_tokens - opt_avg_tokens

        print(f"\nCompression:")
        print(f"  Token savings: {savings:,.0f} tokens")
        print(f"  Compression ratio: {compression:.1f}×")

        # Compare overall scores (failures = wrong)
        overall_diff = opt_overall_score - nat_overall_score
        print(f"\nOverall Score Comparison (failures = wrong):")
        if abs(overall_diff) < 2:
            print(f"  Comparable (Δ{overall_diff:.1f}%)")
        elif overall_diff > 0:
            print(f"  Optical is {overall_diff:.1f}% better overall!")
            print(f"  (Optical handles long documents that cause native OOM/context exceeded)")
        else:
            print(f"  Native is {-overall_diff:.1f}% better overall")


def main():
    parser = argparse.ArgumentParser(description="Universal Optical Compression Testing")

    # Model configuration
    parser.add_argument('--vlm_model_path', type=str, required=True,
                        help='Path to VLM model')
    parser.add_argument('--vlm_type', type=str, default='auto',
                        choices=['auto', 'qwen3', 'llava'],
                        help='VLM type')
    parser.add_argument('--adapter_checkpoint', type=str, required=True,
                        help='Path to trained adapter checkpoint')
    parser.add_argument('--deepencoder_path', type=str, default=None,
                        help='Path to DeepEncoder weights')
    parser.add_argument('--quantization', type=str, default=None,
                        choices=['int4', 'int8'],
                        help='Quantization type for large models (int4 recommended for 8B on 12GB GPU)')

    # Test mode
    parser.add_argument('--text', type=str, default=None,
                        help='Test on custom text')
    parser.add_argument('--file', type=str, default=None,
                        help='Test on text file')
    parser.add_argument('--benchmark', type=str, default=None,
                        choices=['longbench'],
                        help='Run benchmark')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of benchmark samples')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmark (all samples)')
    parser.add_argument('--question', type=str, default=None,
                        help='Question to ask about text/file')

    # Settings
    parser.add_argument('--max_pages', type=int, default=200,
                        help='Max pages to render')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')

    args = parser.parse_args()

    print("="*80)
    print("UNIVERSAL OPTICAL COMPRESSION TESTING")
    print("="*80)

    # Load VLM
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)

    vlm_model, processor = load_vlm_model(args.vlm_model_path, args.vlm_type, args.quantization, args.device)

    # Load OpticalEncoder with trained adapter
    encoder = OpticalEncoder.from_pretrained(
        vlm_model=vlm_model,
        adapter_checkpoint=args.adapter_checkpoint,
        deepencoder_path=args.deepencoder_path,
        max_pages=args.max_pages,
        device=args.device
    )

    print(f"\n✓ OpticalEncoder loaded with trained adapter")
    print(f"  - Target dimension: {encoder.target_dim}")
    print(f"  - Adapter params: {encoder.adapter.get_num_params():,}")

    # Run tests
    if args.benchmark == 'longbench':
        # LongBench v2 benchmark
        results = test_longbench_v2(
            encoder, vlm_model, processor, args.device,
            num_samples=args.num_samples,
            full=args.full
        )
        print_results(results)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")

    elif args.text or args.file:
        # Custom text test
        print("\n" + "="*80)
        print("TESTING CUSTOM TEXT")
        print("="*80)

        if args.file:
            with open(args.file, 'r') as f:
                text = f.read()
            print(f"\n✓ Loaded text from: {args.file}")
        else:
            text = args.text

        print(f"  - Text length: {len(text):,} chars")

        # Test optical
        print("\n[1/2] Testing with optical compression...")
        result_optical = test_optical_compression(
            encoder, text, vlm_model, processor, args.question, args.device, args.max_pages
        )

        print(f"\nOptical Compression:")
        print(f"  Answer: {result_optical['answer'][:200]}...")
        print(f"  Vision tokens: {result_optical['vision_tokens']:,}")
        print(f"  Text tokens: {result_optical['text_tokens']:,}")
        print(f"  Total tokens: {result_optical['total_tokens']:,}")
        print(f"  Pages: {result_optical['num_pages']}")
        print(f"  Time: {result_optical['total_time']:.2f}s")

        # Test native
        print("\n[2/2] Testing with native text...")
        try:
            result_native = test_native_text(
                text, vlm_model, processor, args.question, args.device
            )

            print(f"\nNative Text:")
            print(f"  Answer: {result_native['answer'][:200]}...")
            print(f"  Total tokens: {result_native['total_tokens']:,}")
            print(f"  Time: {result_native['gen_time']:.2f}s")
            print(f"  Truncated: {result_native['truncated']}")

            # Comparison
            compression = result_native['total_tokens'] / result_optical['total_tokens']
            print(f"\nCompression: {compression:.1f}×")

        except Exception as e:
            print(f"\n⚠️  Native text failed (expected for long contexts): {e}")

    else:
        print("\n❌ ERROR: Must specify --text, --file, or --benchmark")
        print("\nExamples:")
        print("  # Custom text")
        print("  python test.py --vlm_model_path /path/to/model \\")
        print("                 --adapter_checkpoint adapters/adapter.pth \\")
        print("                 --text 'Your text here'")
        print("\n  # LongBench v2")
        print("  python test.py --vlm_model_path /path/to/model \\")
        print("                 --adapter_checkpoint adapters/adapter.pth \\")
        print("                 --benchmark longbench --num_samples 20")

    print("\n" + "="*80)
    print("✓ TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
