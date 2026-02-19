"""
Thai OCR Web Application
Professional web interface for Thai handwriting recognition
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64

from model_utils import get_model

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'best_model.pt')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'Model_Implement', 'thai_sp_30000.model')

# Global model
model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_model():
    """Initialize model (lazy loading)"""
    global model
    if model is None:
        print(f"[App] Model path: {MODEL_PATH}")
        print(f"[App] Tokenizer path: {TOKENIZER_PATH}")
        model = get_model(MODEL_PATH, TOKENIZER_PATH)
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle OCR prediction"""
    try:
        ocr_model = init_model()

        # Handle URL request
        if request.is_json:
            data = request.get_json()

            # URL input
            if data.get('url'):
                try:
                    result = ocr_model.predict_from_url(data['url'])
                    return jsonify({
                        'success': True,
                        'text': result,
                        'source': 'url'
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Cannot load image from URL: {str(e)}'
                    }), 400

            # Base64 input
            if data.get('base64'):
                try:
                    base64_data = data['base64']
                    if ',' in base64_data:
                        base64_data = base64_data.split(',')[1]
                    image_bytes = base64.b64decode(base64_data)
                    result = ocr_model.predict_from_bytes(image_bytes)
                    return jsonify({
                        'success': True,
                        'text': result,
                        'source': 'base64'
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Cannot decode base64: {str(e)}'
                    }), 400

            # Sample image input
            if data.get('sample'):
                try:
                    sample_path = data['sample']
                    # Security: only allow files from static/sample_images
                    if not sample_path.startswith('/static/sample_images/'):
                        raise ValueError('Invalid sample path')

                    # Get actual file path
                    file_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        sample_path.lstrip('/')
                    )

                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f'Sample not found: {sample_path}')

                    with open(file_path, 'rb') as f:
                        image_bytes = f.read()

                    result = ocr_model.predict_from_bytes(image_bytes)
                    return jsonify({
                        'success': True,
                        'text': result,
                        'source': 'sample'
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Cannot load sample image: {str(e)}'
                    }), 400

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400

            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400

            try:
                image_bytes = file.read()
                result = ocr_model.predict_from_bytes(image_bytes)
                return jsonify({
                    'success': True,
                    'text': result,
                    'source': 'upload'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Cannot process image: {str(e)}'
                }), 400

        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/model-info')
def model_info():
    try:
        ocr_model = init_model()
        return jsonify({
            'success': True,
            'device': str(ocr_model.device),
            'vocab_size': ocr_model.tokenizer.vocab_size,
            'model_type': 'TrOCR-Thai-Handwriting'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Thai OCR Web Application")
    print("  TrOCR-based Thai Handwriting Recognition")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Tokenizer: {TOKENIZER_PATH}")
    print("=" * 60)

    # Check files
    if not os.path.exists(MODEL_PATH):
        print(f"\n[WARNING] Model not found: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        print(f"\n[WARNING] Tokenizer not found: {TOKENIZER_PATH}")

    print("\n  Starting server at: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
