"""
Thai Handwriting OCR - Gradio Interface
For Hugging Face Spaces Deployment
"""

import gradio as gr
import torch
import os
import io
import requests
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    PreTrainedTokenizer,
)
import sentencepiece as spm
from huggingface_hub import hf_hub_download


# ============== Thai Tokenizer ==============
class ThaiTokenizerFixed(PreTrainedTokenizer):
    """Thai SentencePiece Tokenizer"""

    vocab_files_names = {"vocab_file": "spm.model"}

    def __init__(self, vocab_file=None, **kwargs):
        self.vocab_file = vocab_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.vocab_file)

        super().__init__(
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            **kwargs
        )

    @property
    def pad_token_id(self):
        return 0

    @property
    def unk_token_id(self):
        return 1

    @property
    def bos_token_id(self):
        return 2

    @property
    def eos_token_id(self):
        return 3

    @property
    def vocab_size(self):
        return self.sp.vocab_size()

    def get_vocab(self):
        return {self.sp.id_to_piece(i): i for i in range(self.sp.vocab_size())}

    def _tokenize(self, text):
        return self.sp.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp.decode_pieces(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + bos + token_ids_1 + eos


# ============== Model Loading ==============
class ThaiOCRModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Model] Using device: {self.device}")

        # Load image processor
        print("[Model] Loading image processor...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        # Download model and tokenizer from Hugging Face Hub
        print("[Model] Downloading model from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id="waritkan/thai-ocr-model", filename="best_model.pt")
        tokenizer_path = hf_hub_download(repo_id="waritkan/thai-ocr-model", filename="thai_sp_30000.model")

        # Load tokenizer
        print(f"[Model] Loading tokenizer...")
        self.tokenizer = ThaiTokenizerFixed(vocab_file=tokenizer_path)
        print(f"[Model] Tokenizer vocab size: {self.tokenizer.vocab_size}")

        # Load base model
        print("[Model] Loading TrOCR base model...")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        # Resize embeddings
        self.model.decoder.resize_token_embeddings(self.tokenizer.vocab_size)

        # Configure model
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # Configure generation
        self.model.generation_config.max_length = 128
        self.model.generation_config.early_stopping = True
        self.model.generation_config.num_beams = 4
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Load trained weights
        print(f"[Model] Loading trained weights...")
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print("[Model] Model loaded successfully!")

    def predict(self, image):
        """Perform OCR prediction"""
        if image is None:
            return "Please upload an image"

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess
        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        predicted_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        return predicted_text


# ============== Initialize Model ==============
print("Initializing Thai OCR Model...")
model = ThaiOCRModel()


# ============== Gradio Interface ==============
def process_image(image):
    """Process uploaded image"""
    if image is None:
        return "Please upload an image first"
    return model.predict(image)


# Custom CSS for CMU Purple Theme
custom_css = """
:root {
    --primary-dark: #4A154B;
    --primary: #6B2574;
    --primary-light: #8B3A8E;
    --gold: #C9A227;
}

.gradio-container {
    font-family: 'Sarabun', 'Kanit', sans-serif !important;
}

.header-container {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1A0A1F 0%, #2D1233 50%, #3D1A45 100%);
    border-radius: 16px;
    margin-bottom: 20px;
}

.logo-container {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 20px;
}

.logo-container img {
    height: 80px;
    filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.3));
}

.title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #E8D48B 50%, #D4A5D1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.subtitle {
    color: #D4A5D1;
    font-size: 1rem;
}

.team-section {
    background: white;
    border-radius: 12px;
    padding: 16px;
    margin-top: 20px;
}

.team-title {
    color: #6B2574;
    font-weight: 600;
    margin-bottom: 12px;
}

.team-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.team-member {
    text-align: center;
    padding: 8px;
    background: linear-gradient(135deg, rgba(107, 37, 116, 0.05), rgba(155, 77, 150, 0.08));
    border-radius: 8px;
}

.member-id {
    font-size: 0.75rem;
    color: #C9A227;
    font-weight: 600;
}

.member-name {
    font-size: 0.85rem;
    color: #2D1B36;
}

footer {
    text-align: center;
    padding: 16px;
    color: #D4A5D1;
    font-size: 0.85rem;
}
"""

# Header HTML (without logos for simpler deployment)
header_html = """
<div class="header-container">
    <h1 class="title">Thai Handwriting OCR</h1>
    <p class="subtitle">Thai Handwriting Recognition using TrOCR Deep Learning</p>
    <p style="color: #7A6B7E; font-size: 0.9rem; margin-top: 4px;">
        Faculty of Science, Chiang Mai University
    </p>
    <p style="color: #C9A227; font-size: 0.85rem; margin-top: 4px;">
        Powered by TrOCR + Thai SentencePiece
    </p>
</div>
"""

# Team section HTML
team_html = """
<div class="team-section">
    <div class="team-title">Development Team</div>
    <div class="team-grid">
        <div class="team-member">
            <div class="member-id">650510707</div>
            <div class="member-name">Thichaiyut</div>
        </div>
        <div class="team-member">
            <div class="member-id">650510731</div>
            <div class="member-name">Prakaidao</div>
        </div>
        <div class="team-member">
            <div class="member-id">650510732</div>
            <div class="member-name">Pongsapat</div>
        </div>
        <div class="team-member">
            <div class="member-id">650510735</div>
            <div class="member-name">Waris</div>
        </div>
    </div>
</div>
"""

# Footer HTML
footer_html = """
<footer>
    <strong>Senior Project 2025</strong><br>
    Data Science, Faculty of Science<br>
    <small>Chiang Mai University | Built with PyTorch & Transformers</small>
</footer>
"""

# Build Gradio Interface
with gr.Blocks(title="Thai Handwriting OCR") as demo:
    gr.HTML(header_html)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload", "clipboard"],
            )

            submit_btn = gr.Button(
                "Start Recognition",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Recognized Text",
                lines=5,
                placeholder="Recognition result will appear here...",
            )

    gr.HTML(team_html)
    gr.HTML(footer_html)

    # Events
    submit_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=output_text,
    )

    image_input.change(
        fn=process_image,
        inputs=image_input,
        outputs=output_text,
    )

# Launch
if __name__ == "__main__":
    demo.launch()
