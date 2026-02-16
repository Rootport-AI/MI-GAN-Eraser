import os
import sys
import tempfile
import argparse
import socket
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import logging
import shutil
import cv2
import numpy as np
import onnxruntime as ort

# ONNX Runtime session
SESSION = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            template_folder='.',
            static_folder='.',
            static_url_path='/static')
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure output directory exists
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define model path
MODEL_PATH = os.path.join(BASE_DIR, 'appfiles', 'migan_pipeline_v2.onnx')


def load_model():
    global SESSION
    if SESSION is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            sys.exit(1)

        SESSION = ort.InferenceSession(
            MODEL_PATH,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Log provider info
        active_providers = SESSION.get_providers()
        logger.info(f"Model loaded: {MODEL_PATH}")
        logger.info(f"Active providers: {active_providers}")
        if 'CUDAExecutionProvider' in active_providers:
            logger.info("Using GPU (CUDA) for inference")
        else:
            logger.info("Using CPU for inference (CUDA not available)")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/process', methods=['POST'])
def process_images():
    start_time = time.time()
    app.logger.info(f"Process started at {start_time}")

    image_file = request.files.get('image')
    mask_file = request.files.get('mask')

    if image_file is None or mask_file is None:
        return jsonify({'error': 'image and mask are required'}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "temp.png")
        mask_path = os.path.join(temp_dir, "temp_mask.png")

        image_file.save(image_path)
        mask_file.save(mask_path)

        # Read image as BGR, convert to RGB
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Read mask as RGBA
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            mask_file.seek(0)
            mask_img = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        if mask_img is None:
            return jsonify({'error': 'Failed to read mask image'}), 400

        if mask_img.ndim == 2 or mask_img.shape[2] < 4:
            return jsonify({'error': 'Mask must be RGBA PNG with alpha channel'}), 400

        # Extract alpha channel and threshold
        alpha = mask_img[:, :, 3]
        # Invert polarity: alpha > 25 -> 0 (mask/erase), alpha <= 25 -> 255 (keep)
        mask_gray = np.where(alpha > 25, 0, 255).astype(np.uint8)

        # Convert to NCHW format with batch dimension for ONNX model
        # image: (H, W, 3) -> (1, 3, H, W)
        image_nchw = np.expand_dims(np.transpose(image_rgb, (2, 0, 1)), axis=0)
        # mask: (H, W) -> (1, 1, H, W)
        mask_nchw = mask_gray[np.newaxis, np.newaxis, :, :]

        logger.info(f"Image shape: {image_nchw.shape}, Mask shape: {mask_nchw.shape}")

        output_dir = app.config['OUTPUT_FOLDER']
        result_filename = "tmpimg.png"
        result_path = os.path.join(output_dir, result_filename)

        try:
            infer_start_time = time.time()

            # ONNX inference
            result_nchw = SESSION.run(
                None,
                {'image': image_nchw, 'mask': mask_nchw}
            )[0]

            infer_end_time = time.time()
            logger.info(f"MI-GAN inference took {infer_end_time - infer_start_time:.3f} seconds")

            # Convert NCHW output to HWC: (1, 3, H, W) -> (H, W, 3), then RGB to BGR
            result_rgb = np.transpose(result_nchw[0], (1, 2, 0))
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, result_bgr)

            if not os.path.exists(result_path):
                logger.error(f"Result file not found: {result_path}")
                return jsonify({'error': 'Result file not generated'}), 500

            total_end_time = time.time()
            logger.info(f"Total process took {total_end_time - start_time:.3f} seconds")

            return jsonify({
                'success': True,
                'result_url': f'/output/{result_filename}'
            })

        except Exception as e:
            logger.exception(f"Error during processing: {str(e)}")
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MI-GAN-Eraser Flask server')
    parser.add_argument(
        '--listen',
        action='store_true',
        help='Bind to 0.0.0.0 so other devices on the same LAN can access the app',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7859,
        help='Port to run the Flask server on (default: 7859)',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode (ignored when --listen is used)',
    )
    args = parser.parse_args()

    host = '0.0.0.0' if args.listen else '127.0.0.1'
    debug = args.debug and not args.listen

    load_model()

    if args.listen:
        local_ip = '127.0.0.1'
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(('8.8.8.8', 80))
                local_ip = sock.getsockname()[0]
        except OSError:
            logger.warning('Could not determine local LAN IP address automatically.')

        logger.info('LAN公開中: 他端末からアクセスできます。')
        logger.info(f'アクセスURL例: http://{local_ip}:{args.port}')
        logger.info('セキュリティのため --listen 指定時は debug=False で起動します。')

    app.run(host=host, port=args.port, debug=debug)
