"""
ค้นหารูปภาพที่มีความแม่นยำสูงจาก Thai Handwriting Dataset
สำหรับใช้เป็น sample images บน website
"""

import os
import torch
from PIL import Image
import sentencepiece as spm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from datasets import load_dataset
from difflib import SequenceMatcher
import shutil

# ========================================
# Configuration
# ========================================
BASE_DIR = r'e:\TrOCR_Antigravity'
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'best_model.pt')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'Model_Implement', 'thai_sp_30000.model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'best_samples')
SAMPLE_DIR = os.path.join(BASE_DIR, 'webapp', 'static', 'sample_images')

# จำนวนรูปที่จะทดสอบ (มากกว่า = หารูปดีได้มากกว่า แต่ใช้เวลานาน)
NUM_IMAGES_TO_TEST = 12000  # ทดสอบ 12000 รูป (ใช้เวลา ~1-2 ชั่วโมง)

# จำนวนรูปที่ต้องการเป็น sample
NUM_BEST_SAMPLES = 15

# ========================================
# Thai Tokenizer
# ========================================
class ThaiTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()
        self.bos_token_id = self.sp.PieceToId('<s>')
        self.eos_token_id = self.sp.PieceToId('</s>')
        self.pad_token_id = self.sp.PieceToId('<pad>') if self.sp.PieceToId('<pad>') != -1 else 0

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [i for i in ids if i not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
        return self.sp.DecodeIds(ids)

# ========================================
# Calculate Accuracy
# ========================================
def calculate_similarity(pred, label):
    """คำนวณความคล้ายคลึงระหว่างข้อความ (0-100%)"""
    return SequenceMatcher(None, pred, label).ratio() * 100

def is_exact_match(pred, label):
    """ตรวจสอบว่าตรงกัน 100% หรือไม่"""
    return pred.strip() == label.strip()

# ========================================
# Main
# ========================================
def main():
    print("="*60)
    print("  ค้นหารูปภาพที่มีความแม่นยำสูง")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/4] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    tokenizer = ThaiTokenizer(TOKENIZER_PATH)
    print(f"  Tokenizer vocab: {tokenizer.vocab_size}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model.decoder.resize_token_embeddings(tokenizer.vocab_size)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    image_processor = ViTImageProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    print("  Model loaded!")

    # Load dataset
    print("\n[2/4] Loading dataset...")
    ds = load_dataset("iapp/thai_handwriting_dataset", split="train")
    print(f"  Total images in dataset: {len(ds)}")

    # Test images
    print(f"\n[3/4] Testing {NUM_IMAGES_TO_TEST} images...")
    results = []

    for i in range(min(NUM_IMAGES_TO_TEST, len(ds))):
        sample = ds[i]
        image = sample['image']
        label = sample['text']  # Dataset uses 'text' not 'label'

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Predict
        pixel_values = image_processor(image, return_tensors='pt').pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                decoder_start_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        pred = tokenizer.decode(generated_ids[0])
        similarity = calculate_similarity(pred, label)
        exact = is_exact_match(pred, label)

        results.append({
            'index': i,
            'label': label,
            'prediction': pred,
            'similarity': similarity,
            'exact_match': exact,
            'image': image
        })

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{NUM_IMAGES_TO_TEST} images...", flush=True)

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Print top results
    print("\n[4/4] Top results:")
    print("-"*60)

    for i, r in enumerate(results[:10]):
        status = "✅ EXACT" if r['exact_match'] else f"📊 {r['similarity']:.1f}%"
        print(f"\n{i+1}. [{status}] Index: {r['index']}")
        print(f"   Label:      {r['label']}")
        print(f"   Prediction: {r['prediction']}")

    # Save best samples
    print(f"\n{'='*60}")
    print(f"Saving {NUM_BEST_SAMPLES} best samples to: {OUTPUT_DIR}")
    print("="*60)

    saved_count = 0
    for i, r in enumerate(results):
        if saved_count >= NUM_BEST_SAMPLES:
            break

        # Skip if prediction is too short
        if len(r['prediction'].strip()) < 3:
            continue

        # Save image
        filename = f"best_sample_{saved_count + 1}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        r['image'].save(filepath, 'JPEG', quality=95)

        print(f"\n{saved_count + 1}. Saved: {filename}")
        print(f"   Similarity: {r['similarity']:.1f}%")
        print(f"   Label: {r['label']}")
        print(f"   Prediction: {r['prediction']}")

        saved_count += 1

    # Ask to copy to sample_images
    print(f"\n{'='*60}")
    print("รูปภาพถูกบันทึกที่: " + OUTPUT_DIR)
    print("="*60)
    print("\nถ้าต้องการใช้เป็น sample บน website:")
    print(f"  1. ดูรูปใน folder: {OUTPUT_DIR}")
    print(f"  2. Copy รูปที่ชอบไป: {SAMPLE_DIR}")
    print("  3. เปลี่ยนชื่อเป็น sample4.jpg, sample5.jpg, sample6.jpg")

if __name__ == "__main__":
    main()
