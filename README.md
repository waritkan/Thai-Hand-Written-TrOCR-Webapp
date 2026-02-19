# Thai Handwriting OCR 🇹🇭

ระบบรู้จำลายมือเขียนภาษาไทยด้วย Deep Learning

## Overview
A Thai handwriting recognition system built from scratch by **fine-tuning TrOCR** (Transformer-based OCR) with a **custom-trained Thai SentencePiece tokenizer** (30,000 vocabulary).

## What We Built

### 🔧 Custom Thai Tokenizer
- Trained **SentencePiece tokenizer** from scratch using Thai text corpus
- **30,000 vocabulary** size optimized for Thai language
- Handles Thai script complexities (vowels, tone marks, consonant clusters)

### 🧠 Fine-tuned TrOCR Model
- Base model: `microsoft/trocr-base-handwritten`
- **Fine-tuned on Thai handwriting dataset** ([iapp/thai_handwriting_dataset](https://huggingface.co/datasets/iapp/thai_handwriting_dataset))
- Replaced original tokenizer with our custom Thai tokenizer
- Trained using PyTorch + Transformers

### 🌐 Web Application
- Built Flask web application from scratch
- Drag & drop image upload
- Real-time OCR prediction
- Deployed at [dsctrocr.college](https://dsctrocr.college)

## Model Performance

Tested on [bypkt/thai_handwritten_datasets](https://huggingface.co/datasets/bypkt/thai_handwritten_datasets)

| Model | CharAcc | SeqMatch |
|-------|---------|----------|
| **ThaiOCR (Ours)** | **0.7416** | **0.6614** |
| EasyOCR | 0.6350 | 0.5230 |
| Tesseract | 0.5012 | 0.4070 |
| OpenThaiGPT | 0.3023 | 0.1654 |

- **CharAcc**: Character-level Accuracy
- **SeqMatch**: Sequence Match (Exact Match Rate)

## Project Structure
```
├── Model_Implement/
│   ├── 1 Create_Corpus.ipynb      # สร้าง corpus สำหรับ train tokenizer
│   ├── 2 Create_Tokenaizer.ipynb  # Train SentencePiece tokenizer
│   ├── 3 Train_Model.ipynb        # Fine-tune TrOCR model
│   ├── thai_sp_30000.model        # Trained tokenizer model
│   └── thai_sp_30000.vocab        # Tokenizer vocabulary
├── webapp/                         # Flask web application
└── for_gradio/                     # Gradio demo (HuggingFace Spaces)
```

## Tech Stack
- **Model**: TrOCR (Vision Encoder-Decoder) - Fine-tuned
- **Tokenizer**: SentencePiece - Custom trained (30K Thai vocab)
- **Training**: PyTorch + HuggingFace Transformers
- **Backend**: Flask
- **Frontend**: HTML/CSS/JavaScript

## Team
**Senior Project 2025** - Data Science, Faculty of Science, Chiang Mai University

| Student ID | Name |
|------------|------|
| 650510707 | ธิชัยยุทธ์ ธนะภาษี |
| 650510731 | ประกายดาว พลานามัย |
| 650510732 | พงศพัศ แสงแก้ว |
| 650510735 | วริศ ศิริโฆษิตยางกูร |

## Note
Large files (`best_model.pt` ~1.3GB, `thai_corpus.txt` ~1.7GB) are not included due to GitHub size limits.
