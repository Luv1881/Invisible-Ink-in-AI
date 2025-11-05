"""
Web Interface for Neural Network Steganography
Embedding and Extracting Text in Neural Networks

This Flask application provides a user-friendly web interface for:
- Embedding text messages into neural network weights
- Extracting hidden text from watermarked models
- Viewing embedding statistics and survival rates
"""

import os
import sys
import json
import torch
import torch.nn as nn
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.embedder import NeuralEmbedder
from core.extractor import NeuralExtractor
from core.attacks import TransformationSimulator
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

app = Flask(__name__)
app.secret_key = 'neural_stego_secret_key_2025'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(__file__), 'models')
app.config['METADATA_FOLDER'] = os.path.join(os.path.dirname(__file__), 'metadata')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['METADATA_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_TEXT_EXTENSIONS = {'txt', 'json', 'py', 'java', 'cpp', 'c', 'js', 'html', 'css', 'md', 'xml', 'csv'}
ALLOWED_MODEL_EXTENSIONS = {'pth', 'pt'}

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_cifar10_dataloader(batch_size=128, num_workers=2):
    """
    Load CIFAR-10 dataset for model evaluation.
    Uses standard normalization.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Download CIFAR-10 to parent data directory
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_root, exist_ok=True)

    try:
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testloader
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/embed')
def embed_page():
    """Embedding page"""
    return render_template('embed.html')

@app.route('/extract')
def extract_page():
    """Extraction page"""
    return render_template('extract.html')

@app.route('/api/embed', methods=['POST'])
def api_embed():
    """
    API endpoint for embedding text into neural network.

    Accepts:
    - text: Text content (form field or file upload)
    - redundancy: Redundancy factor (default: 25)
    - step_multiplier: QIM step size (default: 0.25)
    - adaptive: Enable adaptive redundancy (default: true)

    Returns:
    - model_id: Unique identifier for watermarked model
    - statistics: Embedding statistics
    """
    try:
        # Get text from form or file
        text_content = None

        if 'text_file' in request.files:
            file = request.files['text_file']
            if file and file.filename and allowed_file(file.filename, ALLOWED_TEXT_EXTENSIONS):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Read text content
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()

                # Clean up uploaded file
                os.remove(filepath)

        if text_content is None and 'text_content' in request.form:
            text_content = request.form['text_content']

        if not text_content:
            return jsonify({'error': 'No text content provided'}), 400

        # Get embedding parameters
        redundancy = int(request.form.get('redundancy', 25))
        step_multiplier = float(request.form.get('step_multiplier', 0.25))
        adaptive = request.form.get('adaptive', 'true').lower() == 'true'
        validate_quant = request.form.get('validate_quant', 'true').lower() == 'true'

        # Convert text to bytes
        payload = text_content.encode('utf-8')

        print(f"[WEB] Embedding {len(payload)} bytes ({len(text_content)} characters)")
        print(f"[WEB] Parameters: redundancy={redundancy}, step={step_multiplier}, adaptive={adaptive}")

        # Initialize embedder
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedder = NeuralEmbedder(
            model_name='resnet18',
            device=device,
            desired_redundancy=redundancy,
            qim_step_multiplier=step_multiplier,
            adaptive_redundancy=adaptive,
            quantization_simulation=validate_quant,
            accuracy_threshold=0.02
        )

        # Get CIFAR-10 dataloader
        dataloader = get_cifar10_dataloader(batch_size=128, num_workers=0)
        if dataloader is None:
            return jsonify({'error': 'Failed to load CIFAR-10 dataset'}), 500

        # Embed payload
        print("[WEB] Starting embedding process...")
        watermarked_model, metadata = embedder.embed(payload, dataloader)

        # Generate unique model ID
        model_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save watermarked model
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'model_{model_id}.pth')
        torch.save(watermarked_model.state_dict(), model_path)

        # Save metadata (excluding large objects for web storage)
        metadata_web = {
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text_content),
            'payload_size_bytes': len(payload),
            'payload_size_bits': metadata['payload_size_bits'],
            'redundancy_factor': metadata['redundancy_factor'],
            'capacity_percent': metadata['capacity_percent'],
            'baseline_accuracy': metadata['baseline_accuracy'],
            'embedded_accuracy': metadata['embedded_accuracy'],
            'accuracy_drop': metadata['accuracy_drop'],
            'quantization_survival_8bit': metadata.get('quantization_survival_8bit', 0.0),
            'quantization_survival_4bit': metadata.get('quantization_survival_4bit', 0.0),
            'qim_step_multiplier': metadata['qim_step_multiplier'],
            'embedding_seed': metadata['embedding_seed'],
            'whitening_seed': metadata['whitening_seed'],
            'adaptive_redundancy': metadata['adaptive_redundancy']
        }

        # Save full metadata for extraction
        metadata_path = os.path.join(app.config['METADATA_FOLDER'], f'metadata_{model_id}.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        # Save web-friendly metadata
        metadata_json_path = os.path.join(app.config['METADATA_FOLDER'], f'metadata_{model_id}.json')
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata_web, f, indent=2)

        print(f"[WEB] Embedding complete! Model ID: {model_id}")

        return jsonify({
            'success': True,
            'model_id': model_id,
            'statistics': metadata_web
        })

    except Exception as e:
        print(f"[WEB] Error during embedding: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """
    API endpoint for extracting text from watermarked model.

    Accepts:
    - model_id: Model identifier (or upload model file)
    - attack_type: Optional attack simulation (none, quant8, quant4)

    Returns:
    - text: Extracted text content
    - statistics: Extraction statistics
    """
    try:
        model_id = None
        model_path = None
        metadata_path = None

        # Check if model file was uploaded
        if 'model_file' in request.files:
            file = request.files['model_file']
            if file and file.filename and allowed_file(file.filename, ALLOWED_MODEL_EXTENSIONS):
                # Save uploaded model
                model_id = datetime.now().strftime('%Y%m%d_%H%M%S_uploaded')
                filename = f'model_{model_id}.pth'
                model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
                file.save(model_path)

        # Check if metadata file was uploaded
        if 'metadata_file' in request.files:
            file = request.files['metadata_file']
            if file and file.filename:
                # Save uploaded metadata
                if model_id is None:
                    model_id = datetime.now().strftime('%Y%m%d_%H%M%S_uploaded')
                filename = f'metadata_{model_id}.pkl'
                metadata_path = os.path.join(app.config['METADATA_FOLDER'], filename)
                file.save(metadata_path)

        # Or get model_id from form
        if model_id is None and 'model_id' in request.form:
            model_id = request.form['model_id']
            model_path = os.path.join(app.config['MODEL_FOLDER'], f'model_{model_id}.pth')
            metadata_path = os.path.join(app.config['METADATA_FOLDER'], f'metadata_{model_id}.pkl')

        if not model_id or not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return jsonify({'error': 'Model or metadata not found. Please upload both files.'}), 400

        print(f"[WEB] Extracting from model: {model_id}")

        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from torchvision import models
        watermarked_model = models.resnet18(weights=None)
        # Adapt for CIFAR-10
        watermarked_model.fc = nn.Linear(watermarked_model.fc.in_features, 10)
        watermarked_model.load_state_dict(torch.load(model_path, map_location=device))
        watermarked_model.to(device)

        # Apply attack if requested
        attack_type = request.form.get('attack_type', 'none')
        if attack_type == 'quant8':
            print("[WEB] Applying 8-bit quantization attack...")
            attack_simulator = TransformationSimulator(device=device)
            watermarked_model = attack_simulator.quantize_8bit(watermarked_model)
        elif attack_type == 'quant4':
            print("[WEB] Applying 4-bit quantization attack...")
            attack_simulator = TransformationSimulator(device=device)
            watermarked_model = attack_simulator.quantize_4bit(watermarked_model)

        # Extract payload
        extractor = NeuralExtractor(device=device)
        recovered_bytes, stats = extractor.extract(watermarked_model, metadata)

        # Decode text
        try:
            recovered_text = recovered_bytes.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"[WEB] Error decoding text: {e}")
            recovered_text = recovered_bytes.decode('utf-8', errors='ignore')

        # Compute survival rate if original text is in metadata
        survival_rate = None
        if 'payload_size_bits' in metadata:
            original_size = metadata['payload_size_bits']
            extracted_size = len(recovered_bytes) * 8
            if original_size > 0:
                survival_rate = min(extracted_size / original_size, 1.0)

        print(f"[WEB] Extraction complete! Recovered {len(recovered_text)} characters")

        return jsonify({
            'success': True,
            'text': recovered_text,
            'statistics': {
                'extracted_bytes': len(recovered_bytes),
                'extracted_chars': len(recovered_text),
                'mean_confidence': stats.get('mean_confidence', 0.0),
                'min_confidence': stats.get('min_confidence', 0.0),
                'survival_rate': survival_rate,
                'attack_applied': attack_type
            }
        })

    except Exception as e:
        print(f"[WEB] Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_model/<model_id>')
def download_model(model_id):
    """Download watermarked model"""
    model_path = os.path.join(app.config['MODEL_FOLDER'], f'model_{model_id}.pth')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True, download_name=f'watermarked_model_{model_id}.pth')
    return jsonify({'error': 'Model not found'}), 404

@app.route('/api/download_metadata/<model_id>')
def download_metadata(model_id):
    """Download metadata"""
    metadata_path = os.path.join(app.config['METADATA_FOLDER'], f'metadata_{model_id}.pkl')
    if os.path.exists(metadata_path):
        return send_file(metadata_path, as_attachment=True, download_name=f'metadata_{model_id}.pkl')
    return jsonify({'error': 'Metadata not found'}), 404

@app.route('/api/models')
def list_models():
    """List all available models"""
    models = []
    for filename in os.listdir(app.config['METADATA_FOLDER']):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(app.config['METADATA_FOLDER'], filename), 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)
            except:
                pass

    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    return jsonify({'models': models})

if __name__ == '__main__':
    print("=" * 80)
    print("Neural Network Steganography - Web Interface")
    print("=" * 80)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Model folder: {app.config['MODEL_FOLDER']}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    print("\nStarting Flask server...")
    print("Access the web interface at: http://localhost:5000")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
