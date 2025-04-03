# Steganography Detector & Extractor

A tool to detect and extract hidden data from media files.

## Installation

### Prerequisites

- Python 3.7 or higher
- FFmpeg (required for audio and video analysis)

### Installing Dependencies

1. Run the setup script to install all required dependencies:

```bash
python install_requirements.py
```

This script will:
- Install all required Python libraries
- Check if FFmpeg is installed and provide installation instructions if it's not

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install numpy opencv-python Pillow scipy pydub librosa matplotlib
```

### Installing FFmpeg

FFmpeg is required for audio and video analysis.

#### Windows

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or [GitHub Releases](https://github.com/BtbN/FFmpeg-Builds/releases)
2. Extract the zip file to a folder (e.g., C:\ffmpeg)
3. Add the bin folder to your PATH:
   - Right-click on 'This PC' and select 'Properties'
   - Click on 'Advanced system settings'
   - Click on 'Environment Variables'
   - Under 'System variables', find 'Path' and click 'Edit'
   - Click 'New' and add the path to the bin folder (e.g., C:\ffmpeg\bin)
   - Click 'OK' on all dialogs

#### macOS

Using Homebrew:
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

## Usage

Run the application:

```bash
python steg_detector.py
```

1. Click 'Browse' to select a file
2. Click 'Analyze' to detect steganography
3. View the results in the Summary, Details, and Visualization tabs

## Features

- Detection of steganography in images, audio, and video files
- Multiple detection techniques: LSB, DCT, Phase coding, Echo hiding
- Data extraction capabilities
- Visualization of analysis results
