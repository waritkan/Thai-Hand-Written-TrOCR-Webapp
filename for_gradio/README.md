---
title: Thai Handwriting OCR
emoji: ✍️
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Thai Handwriting OCR

ระบบรู้จำลายมือเขียนภาษาไทยด้วยปัญญาประดิษฐ์
Thai Handwriting Recognition using TrOCR Deep Learning

## Features

- Upload image of Thai handwriting
- AI-powered text recognition
- Based on TrOCR architecture with Thai SentencePiece tokenizer

## Model

- **Base Model**: microsoft/trocr-base-handwritten
- **Fine-tuned on**: Thai handwriting dataset
- **Tokenizer**: Thai SentencePiece (30,000 vocab)

## Development Team

| Student ID | Name |
|------------|------|
| 650510707 | ธิชัยยุทธ์ ธนะภาษี |
| 650510731 | ประกายดาว พลานามัย |
| 650510732 | พงศพัศ แสงแก้ว |
| 650510735 | วริศ ศิริโฆษิตยางกูร |

## Institution

**Senior Project 2025**
Data Science, Faculty of Science
Chiang Mai University, Thailand

## Tech Stack

- PyTorch
- Transformers (Hugging Face)
- Gradio
- SentencePiece
