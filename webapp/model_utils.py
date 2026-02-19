"""
Thai OCR Model Utilities
Handles model loading, tokenizer, and image preprocessing
Based on TrOCR architecture with Thai SentencePiece tokenizer
"""

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
import shutil


class ThaiTokenizerFixed(PreTrainedTokenizer):
    """Thai SentencePiece Tokenizer with proper special token handling"""

    vocab_files_names = {"vocab_file": "spm.model"}

    def __init__(self, vocab_file=None, **kwargs):
        self.vocab_file = vocab_file or 'thai_sp_30000.model'
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

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + [1] + ([0] * len(token_ids_1)) + [1]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        out_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "spm.model"
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_file):
            shutil.copy(self.vocab_file, out_file)
        return (out_file,)


class ThaiOCRModel:
    """Thai OCR Model wrapper for inference"""

    def __init__(self, model_path, tokenizer_path, device=None):
        """
        Initialize Thai OCR model

        Args:
            model_path: Path to best_model.pt
            tokenizer_path: Path to thai_sp_30000.model
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Model] Using device: {self.device}")

        # Load image processor from TrOCR
        print("[Model] Loading image processor...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        # Load Thai tokenizer
        print(f"[Model] Loading tokenizer from: {tokenizer_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = ThaiTokenizerFixed(vocab_file=tokenizer_path)
        print(f"[Model] Tokenizer vocab size: {self.tokenizer.vocab_size}")

        # Load base model
        print("[Model] Loading TrOCR base model...")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        # Resize embeddings for Thai tokenizer
        self.model.decoder.resize_token_embeddings(self.tokenizer.vocab_size)

        # Configure model
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # Configure generation settings (new way for transformers 5.x)
        self.model.generation_config.max_length = 128
        self.model.generation_config.early_stopping = True
        self.model.generation_config.num_beams = 4
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Load trained weights
        print(f"[Model] Loading trained weights from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"[Model] Loaded from epoch {checkpoint['epoch'] + 1}")
            if 'val_loss' in checkpoint:
                print(f"[Model] Validation loss: {checkpoint['val_loss']:.4f}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print("[Model] Model loaded successfully!")

    def preprocess_image(self, image):
        """
        Preprocess image for the model

        Args:
            image: PIL Image object

        Returns:
            Preprocessed pixel values tensor
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Use TrOCR image processor
        pixel_values = self.image_processor(
            image,
            return_tensors="pt"
        ).pixel_values

        return pixel_values.to(self.device)

    def load_image_from_url(self, url, timeout=15):
        """Load image from URL"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image

    def load_image_from_bytes(self, image_bytes):
        """Load image from bytes"""
        image = Image.open(io.BytesIO(image_bytes))
        return image

    def predict(self, image):
        """
        Perform OCR prediction

        Args:
            image: PIL Image object

        Returns:
            Predicted text string
        """
        # Preprocess
        pixel_values = self.preprocess_image(image)

        # Generate prediction
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

    def predict_from_url(self, url):
        """Predict from image URL"""
        image = self.load_image_from_url(url)
        return self.predict(image)

    def predict_from_bytes(self, image_bytes):
        """Predict from image bytes"""
        image = self.load_image_from_bytes(image_bytes)
        return self.predict(image)


# Singleton model instance
_model_instance = None


def get_model(model_path=None, tokenizer_path=None):
    """Get or create model instance (singleton)"""
    global _model_instance

    if _model_instance is None:
        if model_path is None or tokenizer_path is None:
            # Default paths
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = model_path or os.path.join(base_dir, 'Model', 'best_model.pt')
            tokenizer_path = tokenizer_path or os.path.join(base_dir, 'Model_Implement', 'thai_sp_30000.model')

        _model_instance = ThaiOCRModel(
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )

    return _model_instance
