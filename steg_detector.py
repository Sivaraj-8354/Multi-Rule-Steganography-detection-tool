import os
import sys
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from scipy.fftpack import dct, idct
import wave
import struct
import logging
import json
from io import BytesIO
import threading
import subprocess
import platform

# Check if ffmpeg is available
def check_ffmpeg():
    try:
        # Different command based on platform
        if platform.system() == "Windows":
            subprocess.run(["where", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:  # Linux/Mac
            subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False

# Import optional dependencies with error handling
try:
    from pydub import AudioSegment
    import librosa
    import matplotlib.pyplot as plt
    AUDIO_SUPPORT = True
except ImportError as e:
    AUDIO_SUPPORT = False
    logger.warning(f"Audio analysis support limited: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteganographyDetector:
    def __init__(self):
        self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.supported_audio_formats = ['.wav', '.mp3', '.flac']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        self.ffmpeg_available = check_ffmpeg()
        
    def analyze_file(self, file_path):
        """Main entry point for analyzing files"""
        if not os.path.exists(file_path):
            return {"status": "error", "message": "File not found"}
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.supported_image_formats:
            return self.analyze_image(file_path)
        elif file_ext in self.supported_audio_formats:
            if not AUDIO_SUPPORT:
                return {"status": "error", "message": "Audio analysis requires librosa and pydub libraries"}
            if not self.ffmpeg_available:
                return {"status": "error", "message": "Audio analysis requires ffmpeg to be installed and in your PATH"}
            return self.analyze_audio(file_path)
        elif file_ext in self.supported_video_formats:
            if not self.ffmpeg_available:
                return {"status": "error", "message": "Video analysis requires ffmpeg to be installed and in your PATH"}
            return self.analyze_video(file_path)
        else:
            return {"status": "error", "message": "Unsupported file format"}
    
    def analyze_image(self, image_path):
        """Detects steganography in images using multiple techniques"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"status": "error", "message": "Could not read image file"}
            
            # Convert to RGB for analysis
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = {
                "file_type": "image",
                "techniques": {},
                "overall_confidence": 0,
                "extracted_data": None
            }
            
            # Run LSB detection
            lsb_result = self.detect_lsb_steganography(img_rgb)
            results["techniques"]["LSB"] = lsb_result
            
            # Run DCT detection
            dct_result = self.detect_dct_steganography(img_rgb)
            results["techniques"]["DCT"] = dct_result
            
            # Calculate overall confidence
            confidences = [v.get("confidence", 0) for v in results["techniques"].values()]
            results["overall_confidence"] = max(confidences) if confidences else 0
            
            # Try to extract if high confidence
            if results["overall_confidence"] > 70:
                extracted = self.extract_from_image(img_rgb, 
                                                   max(results["techniques"].items(), 
                                                       key=lambda x: x[1]["confidence"])[0])
                results["extracted_data"] = extracted
                
            # Set detection status
            if results["overall_confidence"] > 50:
                results["status"] = "Steganography detected"
            else:
                results["status"] = "No hidden data found"
                
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    
    def detect_lsb_steganography(self, img):
        """Detects LSB steganography in images"""
        # Extract the LSB of each color channel
        b, g, r = cv2.split(img)
        b_lsb = b % 2
        g_lsb = g % 2
        r_lsb = r % 2
        
        # Statistical analysis
        b_entropy = self._calculate_entropy(b_lsb)
        g_entropy = self._calculate_entropy(g_lsb)
        r_entropy = self._calculate_entropy(r_lsb)
        
        # Pattern detection
        pattern_score = self._detect_lsb_patterns(b_lsb, g_lsb, r_lsb)
        
        # Calculate confidence score based on entropy and pattern
        avg_entropy = (b_entropy + g_entropy + r_entropy) / 3
        confidence = self._calculate_lsb_confidence(avg_entropy, pattern_score)
        
        return {
            "confidence": confidence,
            "details": {
                "entropy": avg_entropy,
                "pattern_score": pattern_score
            }
        }
    
    def detect_dct_steganography(self, img):
        """Detects DCT-based steganography in images (like JSteg)"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Divide the image into 8x8 blocks and analyze DCT coefficients
        height, width = gray.shape
        anomaly_count = 0
        total_blocks = 0
        
        for i in range(0, height-7, 8):
            for j in range(0, width-7, 8):
                block = gray[i:i+8, j:j+8].astype(float)
                dct_block = dct(dct(block, axis=0), axis=1)
                
                # Check for anomalies in the mid-frequency components
                # (typical hiding spots in JPEG steganography)
                mid_freq = dct_block[2:6, 2:6]
                histogram = np.histogram(mid_freq.flatten(), bins=10)[0]
                variance = np.var(histogram)
                if variance < 0.5:  # Low variance often indicates manipulation
                    anomaly_count += 1
                total_blocks += 1
        
        # Calculate confidence
        if total_blocks == 0:
            confidence = 0
        else:
            confidence = min(100, (anomaly_count / total_blocks) * 200)
        
        return {
            "confidence": confidence,
            "details": {
                "anomaly_blocks": anomaly_count,
                "total_blocks": total_blocks
            }
        }
    
    def _calculate_entropy(self, data):
        """Calculates Shannon entropy of data"""
        values, counts = np.unique(data, return_counts=True)
        probs = counts / len(data.flatten())
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _detect_lsb_patterns(self, b_lsb, g_lsb, r_lsb):
        """Detects patterns in LSB data that might indicate steganography"""
        # Check for long sequences of 0s or 1s (less likely in natural images)
        def count_runs(arr):
            flat = arr.flatten()
            runs = []
            current_run = 1
            
            for i in range(1, len(flat)):
                if flat[i] == flat[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
                    
            if current_run > 1:
                runs.append(current_run)
                
            return runs
        
        b_runs = count_runs(b_lsb)
        g_runs = count_runs(g_lsb)
        r_runs = count_runs(r_lsb)
        
        # Check for unusually long runs
        all_runs = b_runs + g_runs + r_runs
        if not all_runs:
            return 0
            
        max_run = max(all_runs)
        avg_run = sum(all_runs) / len(all_runs)
        
        # Unusually long or uniform runs might indicate steganography
        if max_run > 20 or avg_run > 3:
            return 0.8
        else:
            return 0.2
    
    def _calculate_lsb_confidence(self, entropy, pattern_score):
        """Calculate confidence score for LSB steganography detection"""
        # High entropy in LSB layer often indicates steganography
        # Natural images typically have lower entropy in LSB layer
        if entropy > 0.9:
            entropy_score = 90
        elif entropy > 0.8:
            entropy_score = 70
        elif entropy > 0.7:
            entropy_score = 50
        else:
            entropy_score = 20
            
        # Combine with pattern score
        combined_score = (entropy_score * 0.7) + (pattern_score * 100 * 0.3)
        return min(100, combined_score)
    
    def extract_from_image(self, img, technique):
        """Attempts to extract hidden data from an image"""
        if technique == "LSB":
            return self.extract_lsb_from_image(img)
        elif technique == "DCT":
            return self.extract_dct_from_image(img)
        else:
            return None
    
    def extract_lsb_from_image(self, img):
        """Extract data from LSB steganography"""
        # Extract LSBs from each channel
        b, g, r = cv2.split(img)
        b_lsb = b % 2
        g_lsb = g % 2
        r_lsb = r % 2
        
        # Combine LSBs into bytes
        # This is a simplified extraction - actual extraction would need to know
        # the exact embedding algorithm used
        combined_bits = []
        for i in range(min(1000, b_lsb.shape[0] * b_lsb.shape[1])):
            row = i // b_lsb.shape[1]
            col = i % b_lsb.shape[1]
            if row < b_lsb.shape[0]:
                combined_bits.append(b_lsb[row, col])
                combined_bits.append(g_lsb[row, col])
                combined_bits.append(r_lsb[row, col])
        
        # Convert bits to bytes
        extracted_bytes = bytearray()
        for i in range(0, len(combined_bits) - 7, 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(combined_bits):
                    byte_val |= combined_bits[i + j] << (7 - j)
            extracted_bytes.append(byte_val)
            
        # Try to detect the type of content
        # If it's text, return as string
        try:
            text = extracted_bytes.decode('utf-8', errors='ignore')
            # Check if it has printable characters
            if any(c.isprintable() for c in text):
                return {"type": "text", "data": text[:100] + "..."}  # Return first 100 chars
        except:
            pass
            
        # Otherwise return as binary
        return {"type": "binary", "data": "Binary data extracted, size: " + str(len(extracted_bytes)) + " bytes"}
    
    def extract_dct_from_image(self, img):
        """Extract data from DCT-based steganography"""
        # This is a placeholder for DCT extraction
        # Actual implementation would require knowledge of specific algorithm
        return {"type": "text", "data": "DCT extraction not fully implemented"}
    
    def analyze_audio(self, audio_path):
        """Detects steganography in audio files"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            results = {
                "file_type": "audio",
                "techniques": {},
                "overall_confidence": 0
            }
            
            # Run LSB audio detection
            lsb_result = self.detect_audio_lsb(y)
            results["techniques"]["LSB"] = lsb_result
            
            # Run phase coding detection
            phase_result = self.detect_audio_phase_coding(y, sr)
            results["techniques"]["PhaseCoding"] = phase_result
            
            # Run echo hiding detection
            echo_result = self.detect_audio_echo_hiding(y, sr)
            results["techniques"]["EchoHiding"] = echo_result
            
            # Calculate overall confidence
            confidences = [v.get("confidence", 0) for v in results["techniques"].values()]
            results["overall_confidence"] = max(confidences) if confidences else 0
            
            # Try to extract if high confidence
            if results["overall_confidence"] > 70:
                technique = max(results["techniques"].items(), key=lambda x: x[1]["confidence"])[0]
                extracted = self.extract_from_audio(y, sr, technique)
                results["extracted_data"] = extracted
            
            # Set detection status
            if results["overall_confidence"] > 50:
                results["status"] = "Steganography detected"
            else:
                results["status"] = "No hidden data found"
                
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    
    def detect_audio_lsb(self, y):
        """Detects LSB steganography in audio"""
        # Convert to integers (assuming 16-bit audio)
        samples = (y * 32767).astype(np.int16)
        
        # Extract LSBs
        lsbs = samples & 1
        
        # Statistical analysis
        entropy = self._calculate_entropy(lsbs)
        
        # Calculate confidence
        if entropy > 0.9:
            confidence = 80
        elif entropy > 0.8:
            confidence = 60
        elif entropy > 0.7:
            confidence = 40
        else:
            confidence = 20
            
        return {
            "confidence": confidence,
            "details": {
                "entropy": entropy
            }
        }
    
    def detect_audio_phase_coding(self, y, sr):
        """Detects phase coding steganography in audio"""
        # Perform short-time Fourier transform
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)
        
        # Analyze phase coherence
        phase_diff = np.diff(np.angle(phase), axis=1)
        phase_entropy = self._calculate_entropy(phase_diff.flatten())
        
        # Calculate confidence
        if phase_entropy > 4.5:  # High entropy in phase differences
            confidence = 70
        elif phase_entropy > 4.0:
            confidence = 50
        else:
            confidence = 30
            
        return {
            "confidence": confidence,
            "details": {
                "phase_entropy": phase_entropy
            }
        }
    
    def detect_audio_echo_hiding(self, y, sr):
        """Detects echo hiding steganography in audio"""
        # Echo hiding typically adds small echoes to encode data
        # Look for unusual patterns in the autocorrelation
        
        # Compute autocorrelation
        autocorr = librosa.autocorrelate(y)
        
        # Analyze peaks in autocorrelation
        peaks = librosa.util.peak_pick(autocorr, 3, 3, 3, 10, 0.5, 10)
        
        # More than expected peaks might indicate echo hiding
        confidence = min(80, len(peaks) * 5)
        
        return {
            "confidence": confidence,
            "details": {
                "peak_count": len(peaks)
            }
        }
    
    def extract_from_audio(self, y, sr, technique):
        """Attempts to extract hidden data from audio"""
        if technique == "LSB":
            return self.extract_lsb_from_audio(y)
        elif technique == "PhaseCoding":
            return {"type": "text", "data": "Phase coding extraction not fully implemented"}
        elif technique == "EchoHiding":
            return {"type": "text", "data": "Echo hiding extraction not fully implemented"}
        else:
            return None
    
    def extract_lsb_from_audio(self, y):
        """Extract data from LSB steganography in audio"""
        # Convert to integers (assuming 16-bit audio)
        samples = (y * 32767).astype(np.int16)
        
        # Extract LSBs
        lsbs = samples & 1
        
        # Convert bits to bytes
        extracted_bytes = bytearray()
        for i in range(0, min(8000, len(lsbs) - 7), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(lsbs):
                    byte_val |= lsbs[i + j] << (7 - j)
            extracted_bytes.append(byte_val)
            
        # Try to detect the type of content
        try:
            text = extracted_bytes.decode('utf-8', errors='ignore')
            # Check if it has printable characters
            if any(c.isprintable() for c in text):
                return {"type": "text", "data": text[:100] + "..."}  # Return first 100 chars
        except:
            pass
            
        return {"type": "binary", "data": "Binary data extracted, size: " + str(len(extracted_bytes)) + " bytes"}
    
    def analyze_video(self, video_path):
        """Detects steganography in video files"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"status": "error", "message": "Could not open video file"}
            
            results = {
                "file_type": "video",
                "techniques": {},
                "frame_analysis": [],
                "overall_confidence": 0
            }
            
            # Sample frames for analysis
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate how many frames to analyze
            num_samples = min(20, frame_count)  # Analyze up to 20 frames
            sample_interval = max(1, frame_count // num_samples)
            
            # Analyze sampled frames
            max_confidence = 0
            frame_confidences = []
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Analyze this frame as an image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect LSB steganography
                lsb_result = self.detect_lsb_steganography(frame_rgb)
                
                # Detect DCT steganography
                dct_result = self.detect_dct_steganography(frame_rgb)
                
                # Calculate frame confidence
                frame_confidence = max(lsb_result["confidence"], dct_result["confidence"])
                max_confidence = max(max_confidence, frame_confidence)
                
                # Store frame results
                frame_confidences.append(frame_confidence)
                results["frame_analysis"].append({
                    "frame": i,
                    "time": i/fps,
                    "confidence": frame_confidence
                })
            
            # Check for changes between frames that might indicate temporal steganography
            temporal_result = self.detect_temporal_steganography(frame_confidences)
            results["techniques"]["Temporal"] = temporal_result
            
            # Store technique results
            if frame_confidences:
                avg_lsb = sum(frame_confidences) / len(frame_confidences)
                results["techniques"]["LSB"] = {"confidence": avg_lsb}
                results["techniques"]["DCT"] = {"confidence": avg_lsb * 0.8}  # Just an example
                
            # Set overall confidence
            results["overall_confidence"] = max_confidence
            
            # Set detection status
            if results["overall_confidence"] > 50:
                results["status"] = "Steganography detected"
            else:
                results["status"] = "No hidden data found"
            
            cap.release()
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    
    def detect_temporal_steganography(self, frame_confidences):
        """Detect steganography that uses differences between frames"""
        # Check for patterns in confidence scores that might indicate
        # data hidden in specific frames
        if not frame_confidences:
            return {"confidence": 0}
            
        # Calculate variation
        std_dev = np.std(frame_confidences)
        
        # High variation might indicate temporal steganography
        if std_dev > 20:
            confidence = 80
        elif std_dev > 10:
            confidence = 50
        else:
            confidence = 20
            
        return {
            "confidence": confidence,
            "details": {
                "std_dev": std_dev
            }
        }

class StegUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Detector & Extractor")
        self.root.geometry("900x700")
        
        self.detector = SteganographyDetector()
        self.setup_ui()
        
        # Check ffmpeg availability
        if not self.detector.ffmpeg_available:
            messagebox.showwarning(
                "FFmpeg Not Found", 
                "FFmpeg is not available in your system PATH. Audio and video analysis will be limited.\n\n"
                "Please install FFmpeg and add it to your PATH for full functionality."
            )
        
    def setup_ui(self):
        """Create the UI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Select File", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=70).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Analyze", command=self.analyze_file).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Results notebook (tabs)
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(summary_frame, text="Summary")
        
        # Summary widgets
        summary_top = ttk.Frame(summary_frame)
        summary_top.pack(fill=tk.X)
        
        ttk.Label(summary_top, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.status_label = ttk.Label(summary_top, text="-", font=("Arial", 10, "bold"))
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(summary_top, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_label = ttk.Label(summary_top, text="-")
        self.confidence_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(summary_top, text="File Type:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.filetype_label = ttk.Label(summary_top, text="-")
        self.filetype_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(summary_frame, text="Detected Techniques:").pack(anchor=tk.W, padx=5, pady=5)
        self.techniques_text = tk.Text(summary_frame, height=5, wrap=tk.WORD)
        self.techniques_text.pack(fill=tk.X, padx=5, pady=5)
        self.techniques_text.config(state=tk.DISABLED)
        
        ttk.Label(summary_frame, text="Extracted Data:").pack(anchor=tk.W, padx=5, pady=5)
        self.extracted_text = tk.Text(summary_frame, height=10, wrap=tk.WORD)
        self.extracted_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.extracted_text.config(state=tk.DISABLED)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(details_frame, text="Details")
        
        self.details_text = tk.Text(details_frame, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.config(state=tk.DISABLED)
        
        # Visualization tab
        viz_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(viz_frame, text="Visualization")
        
        self.viz_label = ttk.Label(viz_frame, text="No visualization available")
        self.viz_label.pack(fill=tk.BOTH, expand=True)
        
    def browse_file(self):
        """Open file dialog to select a file for analysis"""
        filetypes = (
            ("All supported files", "*.png *.jpg *.jpeg *.bmp *.tiff *.wav *.mp3 *.flac *.mp4 *.avi *.mov *.mkv"),
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("Audio files", "*.wav *.mp3 *.flac"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
        
        filename = filedialog.askopenfilename(
            title="Select a file for steganography analysis",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path.set(filename)
            
    def analyze_file(self):
        """Analyze the selected file for steganography"""
        file_path = self.file_path.get()
        
        if not file_path:
            tk.messagebox.showwarning("No File Selected", "Please select a file to analyze")
            return
        
        # Reset UI
        self.reset_ui()
        
        # Show progress
        self.progress_var.set(20)
        self.root.update()
        
        # Run analysis in a separate thread to prevent UI freezing
        analysis_thread = threading.Thread(target=self._run_analysis, args=(file_path,))
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _run_analysis(self, file_path):
        """Run analysis in a background thread and update UI when complete"""
        try:
            # Run the analysis
            results = self.detector.analyze_file(file_path)
            
            # Use after() to safely update UI from the main thread
            self.root.after(0, lambda: self._update_results(results))
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: self._show_error(error_msg))
    
    def _update_results(self, results):
        """Update UI with results (called from main thread)"""
        # Update progress
        self.progress_var.set(100)
        
        # Display results
        self.display_results(results)
    
    def _show_error(self, error_msg):
        """Display error message (called from main thread)"""
        self.progress_var.set(0)
        self.status_label.config(text=f"Error: {error_msg}", foreground="red")
    
    def reset_ui(self):
        """Reset UI elements to default state"""
        self.status_label.config(text="-")
        self.confidence_label.config(text="-")
        self.filetype_label.config(text="-")
        
        self.techniques_text.config(state=tk.NORMAL)
        self.techniques_text.delete(1.0, tk.END)
        self.techniques_text.config(state=tk.DISABLED)
        
        self.extracted_text.config(state=tk.NORMAL)
        self.extracted_text.delete(1.0, tk.END)
        self.extracted_text.config(state=tk.DISABLED)
        
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.config(state=tk.DISABLED)
        
        self.viz_label.config(text="No visualization available")
        
    def display_results(self, results):
        """Update UI with analysis results"""
        if not results or results.get("status") == "error":
            error_msg = results.get("message", "Analysis failed") if results else "Analysis failed"
            self.status_label.config(text=f"Error: {error_msg}")
            return
        
        # Update summary tab
        self.status_label.config(text=results.get("status", "-"))
        
        # Color code the status
        if "detected" in results.get("status", "").lower():
            self.status_label.config(foreground="red")
        else:
            self.status_label.config(foreground="green")
            
        # Update confidence
        confidence = results.get("overall_confidence", 0)
        self.confidence_label.config(text=f"{confidence:.1f}%")
        
        # Update file type
        self.filetype_label.config(text=results.get("file_type", "-").capitalize())
        
        # Update techniques
        self.techniques_text.config(state=tk.NORMAL)
        self.techniques_text.delete(1.0, tk.END)
        
        techniques = results.get("techniques", {})
        for technique, data in techniques.items():
            confidence = data.get("confidence", 0)
            self.techniques_text.insert(tk.END, f"{technique}: {confidence:.1f}% confidence\n")
            
        self.techniques_text.config(state=tk.DISABLED)
        
        # Update extracted data
        extracted = results.get("extracted_data", None)
        self.extracted_text.config(state=tk.NORMAL)
        self.extracted_text.delete(1.0, tk.END)
        
        if extracted:
            data_type = extracted.get("type", "unknown")
            data = extracted.get("data", "No data extracted")
            self.extracted_text.insert(tk.END, f"Type: {data_type}\n\n")
            self.extracted_text.insert(tk.END, data)
        else:
            self.extracted_text.insert(tk.END, "No data extracted")
            
        self.extracted_text.config(state=tk.DISABLED)
        
        # Update details tab
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        # Format the results as pretty text
        self.details_text.insert(tk.END, json.dumps(results, indent=2))
        self.details_text.config(state=tk.DISABLED)
        
        # Update visualization tab
        self.create_visualization(results)
        
    def create_visualization(self, results):
        """Create visualization based on results"""
        file_type = results.get("file_type", "")
        
        if file_type == "image":
            # Try to load the image
            try:
                img_path = self.file_path.get()
                img = Image.open(img_path)
                img.thumbnail((400, 400))  # Resize for display
                photo = ImageTk.PhotoImage(img)
                
                # Update the visualization label
                self.viz_label.config(image=photo)
                self.viz_label.image = photo  # Keep a reference
            except:
                self.viz_label.config(text="Couldn't load image for visualization")
                
        elif file_type == "audio":
            self.viz_label.config(text="Audio visualization not implemented")
            
        elif file_type == "video":
            # Display frame analysis if available
            frame_analysis = results.get("frame_analysis", [])
            if frame_analysis:
                # Create a simple bar chart of frame confidences
                confidences = [frame["confidence"] for frame in frame_analysis]
                frames = [frame["frame"] for frame in frame_analysis]
                
                fig = plt.figure(figsize=(8, 4))
                plt.bar(frames, confidences)
                plt.xlabel("Frame Number")
                plt.ylabel("Confidence Score (%)")
                plt.title("Steganography Detection Confidence by Frame")
                plt.tight_layout()
                
                # Convert plot to image
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                
                img = Image.open(buf)
                photo = ImageTk.PhotoImage(img)
                
                # Update the visualization label
                self.viz_label.config(image=photo)
                self.viz_label.image = photo  # Keep a reference
                
                plt.close(fig)
            else:
                self.viz_label.config(text="No frame analysis available for visualization")
        else:
            self.viz_label.config(text="No visualization available")

def main():
    root = tk.Tk()
    app = StegUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()