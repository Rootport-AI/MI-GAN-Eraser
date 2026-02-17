# MI-GAN-Eraser

## Overview
MI-GAN-Eraser is a web-based image editing tool that uses the [MI-GAN inpainting model](https://github.com/Picsart-AI-Research/MI-GAN) (ICCV 2023) to remove unwanted objects from images. Users can draw masks on the image using a brush, eraser, or bucket tool, and then process the image with AI to seamlessly erase the masked areas. This tool is built with Flask, JavaScript, and ONNX Runtime.
![MI-GAN-Eraser Demo](https://raw.githubusercontent.com/Rootport-AI/MI-GAN-Eraser/main/demo.gif)  

## Features
- Image Upload: Load images via file selection or drag-and-drop.  
- Drawing Tools: Brush, eraser, and bucket fill for mask creation.   
- AI Processing: Send masked images to MI-GAN for inpainting.
- Download: Save the processed image locally.

## Installation
### Prerequisites
- Windows OS (tested on Windows 10/11)
- Python 3.10+
- NVIDIA GPU with CUDA support recommended (CPU fallback available)

### Steps
1. Download the Repository   
- Clone or download this repository from GitHub.  
2. Prepare the Files  
- Ensure the following files are in a single folder:  
```
app.py
index.html
main.js
migan_pipeline_v2.onnx
nouislider.min.css
nouislider.min.js
require.txt
run.bat
run_listen.bat
setup.bat
style.css
\images
    defimg.png
    favicon16.png
    favicon32.png
    favicon64.png
    favicon_original.jpg
```
3. Run the Setup Script  
- Double-click setup.bat in the folder.  
- The script will:  
   - Create an \appfiles directory.  
   - Set up a Python virtual environment.
   - Install dependencies from require.txt.
   - Copy the MI-GAN ONNX model to \appfiles.
4. Verify Installation
- After setup completes, check that \appfiles\migan_pipeline_v2.onnx exists.  

## Usage
1. Launch the Application  
- Local only (default): double-click `run.bat` to start the Flask server bound to `127.0.0.1`.
- LAN share: run `run.bat --listen` from Command Prompt (or double-click `run_listen.bat`) to bind to `0.0.0.0`.
2. Access the Tool  
- Local PC: open http://localhost:7859.  
- LAN clients (when `--listen` is enabled): open `http://<host-pc-ip>:7859`.
3. Edit an Image  
- Load an image using the "Load" button (F) or drag-and-drop.  
- Use the brush (B), eraser (E), or bucket (G) to draw a mask.  
- Press "Run" (Enter) to process the image with MI-GAN.  
- Download the result with the "Download" button (S).  

### Startup examples
- `run.bat`
  - Localhost only (`127.0.0.1`), not reachable from other devices.
- `run.bat --listen`
  - LAN accessible (`0.0.0.0`), reachable at `http://<host-pc-ip>:7859`.
  - For safety, Flask debug mode is always disabled when `--listen` is set.

### LAN access notes
- Ensure the host PC and client device are on the same network.
- Allow Python/port `7859` through Windows Defender Firewall if prompted.
- Avoid exposing this port to untrusted/public networks.

## Keyboard Shortcuts  
- Ctrl+Z: Undo
- Ctrl+Y: Redo  
- H: Toggle Pan mode  
- B / P: Brush mode  
- E: Eraser mode  
- G: Bucket mode  
- [ ]: Decrease/Increase brush size  
- "+ / -": Zoom in/out  
- 0: Reset view  
- F: Load file  
- Enter: Run  
- S: Download  

## Dependencies
See require.txt for the full list of Python dependencies. Key libraries include:  
- ONNX Runtime GPU (MIT)  
- Flask (BSD 3-Clause)  
- OpenCV (Apache 2.0)  
- noUISlider (MIT, included as nouislider.min.js and nouislider.min.css)  

## License  
This project is licensed under the MIT License. See below for third-party licenses:  

## Third-Party Licenses  
- MI-GAN: A Simple Baseline for Image Inpainting on Mobile Devices  
    - Copyright (c) 2023 Picsart AI Research (PAIR)  
    - Authors: Andranik Sargsyan, Shant Navasardyan, Xingqian Xu, Humphrey Shi  
    - License: MIT  
    - Source: https://github.com/Picsart-AI-Research/MI-GAN  
    - ONNX pipeline model included as migan_pipeline_v2.onnx  
- Font Awesome: CC BY 4.0 (icons), MIT (CSS)  
    - Source: https://fontawesome.com/  
    - Used via CDN in index.html  
- noUISlider: MIT License  
    - Source: https://refreshless.com/nouislider/  
    - Included as nouislider.min.js and nouislider.min.css  

## Notes
- The Undo/Redo functionality for background edits is still under refinement and may not work as expected in this version.  
- For best results, use small brush strokes to erase objects incrementally rather than masking large areas at once.  

## Contributing  
Feel free to fork this repository, submit issues, or send pull requests. Feedback and contributions are welcome!  

## Acknowledgments  
- [Picsart AI Research (PAIR)](https://github.com/Picsart-AI-Research/MI-GAN) for the MI-GAN model.  
- The open-source community for providing invaluable tools and libraries.
