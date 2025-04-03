#!/usr/bin/env python3
import subprocess
import sys
import platform
import os

def check_pip():
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            subprocess.run(["where", "ffmpeg"], check=True, stdout=subprocess.PIPE)
        else:  # Linux/Mac
            subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False

def install_dependencies():
    print("Checking and installing Python dependencies...")
    
    # List of required packages
    packages = [
        "numpy",
        "opencv-python",
        "Pillow",
        "scipy",
        "pydub",
        "librosa",
        "matplotlib"
    ]
    
    if not check_pip():
        print("Error: pip is not installed or not working properly.")
        return False
    
    # Install each package
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"Successfully installed {package}")
        except subprocess.SubprocessError as e:
            print(f"Failed to install {package}: {e}")
            return False
    
    return True

def install_ffmpeg_instructions():
    system = platform.system()
    
    print("\nFFmpeg Installation Instructions:")
    
    if system == "Windows":
        print("""
1. Download FFmpeg from https://ffmpeg.org/download.html or https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract the zip file to a folder (e.g., C:\\ffmpeg)
3. Add the bin folder to your PATH environment variable:
   - Right-click on 'This PC' or 'My Computer' and select 'Properties'
   - Click on 'Advanced system settings'
   - Click on 'Environment Variables'
   - Under 'System variables', find and select 'Path', then click 'Edit'
   - Click 'New' and add the path to the bin folder (e.g., C:\\ffmpeg\\bin)
   - Click 'OK' on all dialogs to save the changes
4. Restart your command prompt or IDE to apply the changes
""")
    elif system == "Darwin":  # macOS
        print("""
Using Homebrew:
1. Install Homebrew if not already installed: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
2. Then run: brew install ffmpeg

Using MacPorts:
1. Install MacPorts if not already installed
2. Then run: sudo port install ffmpeg
""")
    elif system == "Linux":
        print("""
For Ubuntu/Debian:
  sudo apt update
  sudo apt install ffmpeg

For Fedora:
  sudo dnf install ffmpeg

For CentOS/RHEL:
  sudo yum install epel-release
  sudo yum install ffmpeg
""")
    else:
        print("Please visit https://ffmpeg.org/download.html for instructions on how to install FFmpeg on your system.")

def main():
    print("Steganography Detector & Extractor - Setup")
    print("=========================================\n")
    
    # Check FFmpeg
    print("Checking for FFmpeg...")
    if check_ffmpeg():
        print("✓ FFmpeg is installed and available in PATH.")
    else:
        print("✗ FFmpeg is not available in PATH.")
        install_ffmpeg_instructions()
    
    # Install Python dependencies
    if install_dependencies():
        print("\nAll Python dependencies installed successfully!")
    else:
        print("\nSome Python dependencies could not be installed. Please check the error messages above.")
    
    print("\nSetup completed. You can now run the Steganography Detector & Extractor application.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
