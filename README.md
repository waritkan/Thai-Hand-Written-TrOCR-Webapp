# Thai Handwriting OCR 🇹🇭

ระบบรู้จำลายมือเขียนภาษาไทยด้วย Deep Learning

## Overview
A Thai handwriting recognition system using TrOCR (Transformer-based OCR) with a custom Thai SentencePiece tokenizer (30,000 vocab).

## Features
- 🖼️ Recognizes handwritten Thai text from images
- 🌐 Web application with drag & drop interface
- 🚀 Deployed at [dsctrocr.college](https://dsctrocr.college)

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

## Tech Stack
- **Model**: TrOCR (Vision Encoder-Decoder)
- **Tokenizer**: SentencePiece (Thai 30K vocab)
- **Backend**: Flask + PyTorch
- **Frontend**: HTML/CSS/JavaScript

## Team
Senior Project 2025 - Data Science, Faculty of Science, Chiang Mai University

| Student ID | Name |
|------------|------|
| 650510707 | ธิชัยยุทธ์ ธนะภาษี |
| 650510731 | ประกายดาว พลานามัย |
| 650510732 | พงศพัศ แสงแก้ว |
| 650510735 | วริศ ศิริโฆษิตยางกูร |

## Note
Large files (`best_model.pt`, `thai_corpus.txt`) are not included due to GitHub size limits.
