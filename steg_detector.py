import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from PIL import Image, ImageTk
import librosa
import pywt
from scipy.stats import chi2
import json
import csv
from datetime import datetime
from matplotlib import patches  # Add this import

class SteganographyDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Detection Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        self.file_path = None
        self.file_type = None
        self.detection_results = {}
        
        self.create_gui()
    
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # File selection
        ttk.Label(control_frame, text="Select File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        browse_button = ttk.Button(control_frame, text="Browse", command=self.browse_file)
        browse_button.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # File type selection
        ttk.Label(control_frame, text="File Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_type_var = tk.StringVar()
        file_types = ["Image", "Audio", "Video"]
        file_type_combo = ttk.Combobox(control_frame, textvariable=self.file_type_var, values=file_types, state="readonly", width=15)
        file_type_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        file_type_combo.current(0)
        
        # Detection techniques frame
        techniques_frame = ttk.LabelFrame(control_frame, text="Detection Techniques", padding=10)
        techniques_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        
        # Checkbuttons for techniques
        self.statistical_var = tk.BooleanVar(value=True)
        self.entropy_var = tk.BooleanVar(value=True)
        self.transform_var = tk.BooleanVar(value=True)
        self.metadata_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(techniques_frame, text="Statistical Analysis", variable=self.statistical_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(techniques_frame, text="Entropy Analysis", variable=self.entropy_var).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(techniques_frame, text="Transform Domain", variable=self.transform_var).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(techniques_frame, text="Metadata Inspection", variable=self.metadata_var).grid(row=3, column=0, sticky=tk.W, pady=2)
        
        # Threshold settings
        thresholds_frame = ttk.LabelFrame(control_frame, text="Threshold Settings", padding=10)
        thresholds_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        
        ttk.Label(thresholds_frame, text="Chi-Square Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.chi_threshold_var = tk.DoubleVar(value=0.1)
        ttk.Entry(thresholds_frame, textvariable=self.chi_threshold_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(thresholds_frame, text="Entropy Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.entropy_threshold_var = tk.DoubleVar(value=7.0)
        ttk.Entry(thresholds_frame, textvariable=self.entropy_threshold_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        analyze_button = ttk.Button(button_frame, text="Analyze", command=self.analyze_file)
        analyze_button.grid(row=0, column=0, padx=5)
        
        save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results)
        save_button.grid(row=0, column=1, padx=5)
        
        # Right panel - Display area
        display_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File preview frame
        self.preview_frame = ttk.LabelFrame(display_frame, text="File Preview", padding=10)
        self.preview_frame.pack(fill=tk.X, pady=5)
        
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(pady=5)
        
        # Results notebook
        self.results_notebook = ttk.Notebook(display_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.summary_frame, text="Summary")
        
        # Results text
        self.results_text = tk.Text(self.summary_frame, height=10, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.viz_frame, text="Visualization")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_file(self):
        filetypes = (
            ("All files", "*.*"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("Audio files", "*.mp3 *.wav"),
            ("Video files", "*.mp4 *.avi")
        )
        
        self.file_path = filedialog.askopenfilename(filetypes=filetypes)
        if self.file_path:
            self.status_var.set(f"Selected file: {os.path.basename(self.file_path)}")
            self.display_file_preview()
            
            # Auto-detect file type
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.file_type_var.set("Image")
            elif ext in ['.mp3', '.wav']:
                self.file_type_var.set("Audio")
            elif ext in ['.mp4', '.avi']:
                self.file_type_var.set("Video")
    
    def display_file_preview(self):
        if not self.file_path:
            return
            
        # Clear previous preview
        for widget in self.preview_label.winfo_children():
            widget.destroy()
            
        ext = os.path.splitext(self.file_path)[1].lower()
        
        # Display preview based on file type
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                # Load and resize image for preview
                img = Image.open(self.file_path)
                img.thumbnail((300, 200))
                photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=photo)
                self.preview_label.image = photo  # Keep a reference
            except Exception as e:
                self.preview_label.config(text=f"Could not load image: {e}")
                
        elif ext in ['.mp3', '.wav']:
            # Display audio waveform
            try:
                audio, sr = librosa.load(self.file_path, sr=None, duration=10)
                
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.plot(audio)
                ax.set_title("Audio Waveform (first 10s)")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                
                canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()
                plt.close(fig)
            except Exception as e:
                self.preview_label.config(text=f"Could not load audio: {e}")
                
        elif ext in ['.mp4', '.avi']:
            try:
                # Capture first frame of video
                cap = cv2.VideoCapture(self.file_path)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (300, 200))
                    img = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(img)
                    self.preview_label.config(image=photo)
                    self.preview_label.image = photo  # Keep a reference
                cap.release()
            except Exception as e:
                self.preview_label.config(text=f"Could not load video: {e}")
        else:
            self.preview_label.config(text="File type not supported for preview")
    
    def analyze_file(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        file_type = self.file_type_var.get()
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
            
        self.status_var.set("Analyzing... Please wait")
        self.root.update()
        
        try:
            self.detection_results = {
                "file_name": os.path.basename(self.file_path),
                "file_type": file_type,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "techniques": {},
                "overall_risk": "Low",
                "confidence_score": 0.0
            }
            
            # Run selected analysis techniques
            if file_type == "Image":
                self.analyze_image()
            elif file_type == "Audio":
                self.analyze_audio()
            elif file_type == "Video":
                self.analyze_video()
                
            # Calculate overall risk and confidence
            self.calculate_overall_results()
            
            # Display results
            self.display_results()
            
            self.status_var.set("Analysis complete")
        except Exception as e:
            self.status_var.set("Analysis failed")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def analyze_image(self):
        # Load image
        img = cv2.imread(self.file_path)
        if img is None:
            raise ValueError("Could not load image")
            
        # Statistical Analysis
        if self.statistical_var.get():
            # Histogram analysis
            hist_blue = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_green = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_red = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # Chi-square test on least significant bits
            lsb_array = self.extract_lsbs(img)
            chi_result = self.chi_square_test(lsb_array)
            
            # Store results
            self.detection_results["techniques"]["statistical"] = {
                "chi_square_p_value": chi_result,
                "chi_square_threshold": self.chi_threshold_var.get(),
                "anomaly_detected": chi_result < self.chi_threshold_var.get(),
                "confidence": (1 - chi_result) * 100 if chi_result < 0.5 else 0
            }
            
            # Create histogram visualization
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(hist_blue, color='blue')
            ax.plot(hist_green, color='green')
            ax.plot(hist_red, color='red')
            ax.set_title("Color Histograms")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            plt.close(fig)
            
        # Entropy Analysis
        if self.entropy_var.get():
            entropy_value = self.calculate_entropy(img)
            entropy_threshold = self.entropy_threshold_var.get()
            
            self.detection_results["techniques"]["entropy"] = {
                "entropy_value": entropy_value,
                "entropy_threshold": entropy_threshold,
                "anomaly_detected": entropy_value > entropy_threshold,
                "confidence": (entropy_value / entropy_threshold - 1) * 100 if entropy_value > entropy_threshold else 0
            }
            
        # Transform Domain Analysis
        if self.transform_var.get():
            # DCT analysis
            dct_result = self.analyze_dct(img)
            
            self.detection_results["techniques"]["transform"] = {
                "dct_anomaly_score": dct_result,
                "dct_threshold": 0.2,
                "anomaly_detected": dct_result > 0.2,
                "confidence": dct_result * 100
            }
            
        # Metadata Analysis
        if self.metadata_var.get():
            metadata_result = self.analyze_metadata()
            
            self.detection_results["techniques"]["metadata"] = {
                "suspicious_metadata": metadata_result,
                "anomaly_detected": metadata_result != "None detected",
                "confidence": 70 if metadata_result != "None detected" else 0
            }
        
        # Extract hidden data
        hidden_data = self.extract_hidden_data(img, "image")
        self.detection_results["hidden_data"] = hidden_data
    
    def analyze_audio(self):
        try:
            # Load audio file
            audio, sr = librosa.load(self.file_path, sr=None)
            
            # Statistical Analysis
            if self.statistical_var.get():
                # Calculate audio statistics
                mean = np.mean(audio)
                std = np.std(audio)
                skew = np.mean(((audio - mean)/std)**3) if std > 0 else 0
                
                # Create histogram
                hist, bins = np.histogram(audio, bins=50)
                
                # Detect unusual statistical patterns
                anomaly_score = abs(skew) > 0.5
                
                self.detection_results["techniques"]["statistical"] = {
                    "skewness": skew,
                    "skewness_threshold": 0.5,
                    "anomaly_detected": anomaly_score,
                    "confidence": abs(skew) * 100 if abs(skew) > 0.5 else 0
                }
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(audio, bins=50)
                ax.set_title("Audio Amplitude Histogram")
                ax.set_xlabel("Amplitude")
                ax.set_ylabel("Frequency")
                
                canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                plt.close(fig)
            
            # Entropy Analysis
            if self.entropy_var.get():
                entropy_value = self.calculate_entropy_1d(audio)
                entropy_threshold = 0.9 * self.entropy_threshold_var.get()  # Adjust threshold for audio
                
                self.detection_results["techniques"]["entropy"] = {
                    "entropy_value": entropy_value,
                    "entropy_threshold": entropy_threshold,
                    "anomaly_detected": entropy_value > entropy_threshold,
                    "confidence": (entropy_value / entropy_threshold - 1) * 100 if entropy_value > entropy_threshold else 0
                }
            
            # Transform Domain Analysis
            if self.transform_var.get():
                # FFT analysis
                fft_result = np.abs(np.fft.rfft(audio))
                fft_anomaly = self.detect_fft_anomalies(fft_result)
                
                self.detection_results["techniques"]["transform"] = {
                    "fft_anomaly_score": fft_anomaly,
                    "fft_threshold": 0.3,
                    "anomaly_detected": fft_anomaly > 0.3,
                    "confidence": fft_anomaly * 100
                }
                
                # Create FFT visualization
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(fft_result[:len(fft_result)//10])  # Plot only lower frequencies for clarity
                ax.set_title("Audio Frequency Spectrum")
                ax.set_xlabel("Frequency Bin")
                ax.set_ylabel("Magnitude")
                
                canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                plt.close(fig)
            
            # Metadata Analysis
            if self.metadata_var.get():
                metadata_result = self.analyze_metadata()
                
                self.detection_results["techniques"]["metadata"] = {
                    "suspicious_metadata": metadata_result,
                    "anomaly_detected": metadata_result != "None detected",
                    "confidence": 70 if metadata_result != "None detected" else 0
                }
            
            # Extract hidden data
            hidden_data = self.extract_hidden_data(audio, "audio")
            self.detection_results["hidden_data"] = hidden_data
                
        except Exception as e:
            raise ValueError(f"Audio analysis failed: {str(e)}")
    
    def analyze_video(self):
        try:
            # Open video file
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for analysis
            frame_samples = []
            sample_indices = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_samples.append(frame)
            
            cap.release()
            
            if not frame_samples:
                raise ValueError("Could not extract frames from video")
                
            # Statistical Analysis
            if self.statistical_var.get():
                # Frame difference analysis
                diff_scores = []
                for i in range(1, len(frame_samples)):
                    diff = cv2.absdiff(frame_samples[i], frame_samples[i-1])
                    diff_score = np.mean(diff)
                    diff_scores.append(diff_score)
                
                avg_diff = np.mean(diff_scores) if diff_scores else 0
                std_diff = np.std(diff_scores) if diff_scores else 0
                
                anomaly_score = std_diff / (avg_diff + 1e-10)  # Avoid division by zero
                
                self.detection_results["techniques"]["statistical"] = {
                    "frame_diff_variability": anomaly_score,
                    "variability_threshold": 0.4,
                    "anomaly_detected": anomaly_score > 0.4,
                    "confidence": anomaly_score * 100 if anomaly_score > 0.4 else 0
                }
                
                # Create visualization
                if diff_scores:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(diff_scores)
                    ax.set_title("Frame Difference Scores")
                    ax.set_xlabel("Frame Pair")
                    ax.set_ylabel("Difference Score")
                    
                    canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    plt.close(fig)
            
            # Entropy Analysis
            if self.entropy_var.get():
                entropies = [self.calculate_entropy(frame) for frame in frame_samples]
                avg_entropy = np.mean(entropies)
                entropy_threshold = self.entropy_threshold_var.get() * 0.95  # Adjusted for video
                
                self.detection_results["techniques"]["entropy"] = {
                    "entropy_value": avg_entropy,
                    "entropy_threshold": entropy_threshold,
                    "anomaly_detected": avg_entropy > entropy_threshold,
                    "confidence": (avg_entropy / entropy_threshold - 1) * 100 if avg_entropy > entropy_threshold else 0
                }
            
            # Transform Domain Analysis
            if self.transform_var.get():
                # DCT analysis on a sample of frames
                dct_scores = [self.analyze_dct(frame) for frame in frame_samples]
                avg_dct_score = np.mean(dct_scores)
                
                self.detection_results["techniques"]["transform"] = {
                    "dct_anomaly_score": avg_dct_score,
                    "dct_threshold": 0.25,
                    "anomaly_detected": avg_dct_score > 0.25,
                    "confidence": avg_dct_score * 100
                }
            
            # Metadata Analysis
            if self.metadata_var.get():
                metadata_result = self.analyze_metadata()
                
                self.detection_results["techniques"]["metadata"] = {
                    "suspicious_metadata": metadata_result,
                    "anomaly_detected": metadata_result != "None detected",
                    "confidence": 70 if metadata_result != "None detected" else 0
                }
            
            # Extract hidden data
            hidden_data = self.extract_hidden_data(frame_samples, "video")
            self.detection_results["hidden_data"] = hidden_data
                
        except Exception as e:
            raise ValueError(f"Video analysis failed: {str(e)}")
    
    def extract_lsbs(self, img):
        """Extract least significant bits from image"""
        # Convert to grayscale for simplicity
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # Extract LSBs
        lsb_array = img_gray % 2
        return lsb_array
    
    def chi_square_test(self, lsb_array):
        """Perform chi-square test on LSB array"""
        # Count occurrences of 0s and 1s
        zeros = np.sum(lsb_array == 0)
        ones = np.sum(lsb_array == 1)
        
        # Expected counts (should be approximately equal)
        total = zeros + ones
        expected = total / 2
        
        # Chi-square statistic
        chi_stat = ((zeros - expected)**2 + (ones - expected)**2) / expected
        
        # P-value
        p_value = 1 - chi2.cdf(chi_stat, 1)
        return p_value
    
    def calculate_entropy(self, img):
        """Calculate entropy of an image"""
        # Convert to grayscale for simplicity
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # Calculate histogram
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small constant to avoid log(0)
        return entropy
    
    def calculate_entropy_1d(self, signal):
        """Calculate entropy of 1D signal (e.g., audio)"""
        # Normalize and bin the signal
        signal = signal - np.min(signal)
        if np.max(signal) > 0:
            signal = signal / np.max(signal)
            
        # Create histogram with 256 bins
        hist, _ = np.histogram(signal, bins=256, range=(0, 1), density=True)
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small constant to avoid log(0)
        return entropy
    
    def analyze_dct(self, img):
        """Analyze DCT coefficients for steganography detection"""
        # Convert to grayscale for simplicity
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # Apply DCT to 8x8 blocks
        h, w = img_gray.shape
        anomaly_score = 0
        block_count = 0
        
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = img_gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Analyze coefficients
                # Focus on low-frequency coefficients (excluding DC)
                low_freq = dct_block[0:4, 0:4].flatten()[1:]
                high_freq = dct_block[4:8, 4:8].flatten()
                
                # Calculate ratio
                low_mean = np.mean(np.abs(low_freq))
                high_mean = np.mean(np.abs(high_freq))
                
                if low_mean > 0:
                    ratio = high_mean / low_mean
                    # Unusual ratio indicates potential steganography
                    if ratio > 0.2:  # Threshold based on experimentation
                        anomaly_score += 1
                
                block_count += 1
        
        # Normalize score
        if block_count > 0:
            anomaly_score /= block_count
            
        return anomaly_score
    
    def detect_fft_anomalies(self, fft_result):
        """Detect anomalies in FFT spectrum for audio steganography"""
        # Focus on high-frequency components
        high_freq = fft_result[len(fft_result)//2:]
        
        # Calculate ratio of high to low frequencies
        low_freq = fft_result[:len(fft_result)//2]
        
        high_mean = np.mean(high_freq)
        low_mean = np.mean(low_freq)
        
        if low_mean > 0:
            ratio = high_mean / low_mean
            # Unusual ratio may indicate steganography
            anomaly_score = min(1.0, ratio / 2)  # Normalize to [0, 1]
        else:
            anomaly_score = 0
            
        return anomaly_score
    
    def analyze_metadata(self):
        """Analyze file metadata for suspicious patterns"""
        # This is a simplified implementation
        # In a real-world scenario, you'd use specific libraries for each file type
        
        try:
            # Basic file size check
            file_size = os.path.getsize(self.file_path)
            
            # Check for unusually large files relative to content
            ext = os.path.splitext(self.file_path)[1].lower()
            
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = Image.open(self.file_path)
                width, height = img.size
                expected_size = width * height * 3 / 8  # Rough estimate for compressed images
                
                if file_size > expected_size * 5:
                    return "File size is unusually large for the image dimensions"
                    
            elif ext in ['.mp3', '.wav']:
                audio, sr = librosa.load(self.file_path, sr=None)
                duration = len(audio) / sr
                expected_size = sr * duration * 2 / 8  # Rough estimate for audio (16-bit stereo)
                
                if file_size > expected_size * 10:
                    return "File size is unusually large for the audio duration"
                    
            # Check for unusual file extension vs content type
            # This would require more sophisticated content type detection
            
            return "None detected"
        except Exception as e:
            return f"Error analyzing metadata: {e}"
    
    def calculate_overall_results(self):
        """Calculate overall risk and confidence score"""
        techniques = self.detection_results.get("techniques", {})
        
        # Count anomalies and their confidence scores
        anomaly_count = 0
        total_confidence = 0
        max_confidence = 0
        
        for technique, results in techniques.items():
            if results.get("anomaly_detected", False):
                anomaly_count += 1
                confidence = results.get("confidence", 0)
                total_confidence += confidence
                max_confidence = max(max_confidence, confidence)
        
        # Calculate overall confidence score (weighted average)
        if anomaly_count > 0:
            avg_confidence = total_confidence / anomaly_count
            # Weight towards the highest confidence detection
            overall_confidence = (avg_confidence + max_confidence) / 2
        else:
            overall_confidence = 0
            
        # Determine risk level
        if anomaly_count == 0:
            risk_level = "Low"
        elif anomaly_count == 1:
            risk_level = "Low" if overall_confidence < 50 else "Medium"
        elif anomaly_count == 2:
            risk_level = "Medium" if overall_confidence < 70 else "High"
        else:
            risk_level = "High"
            
        # Update detection results
        self.detection_results["overall_risk"] = risk_level
        self.detection_results["confidence_score"] = overall_confidence
        self.detection_results["anomaly_count"] = anomaly_count
    
    def display_results(self):
        """Display analysis results in the GUI"""
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Display summary
        self.results_text.insert(tk.END, f"File: {self.detection_results['file_name']}\n")
        self.results_text.insert(tk.END, f"Type: {self.detection_results['file_type']}\n")
        self.results_text.insert(tk.END, f"Analysis Date: {self.detection_results['analysis_date']}\n\n")
        
        self.results_text.insert(tk.END, f"RESULTS SUMMARY\n")
        self.results_text.insert(tk.END, f"{'='*30}\n")
        self.results_text.insert(tk.END, f"Overall Risk Level: {self.detection_results['overall_risk']}\n")
        self.results_text.insert(tk.END, f"Confidence Score: {self.detection_results['confidence_score']:.1f}%\n")
        self.results_text.insert(tk.END, f"Anomalies Detected: {self.detection_results.get('anomaly_count', 0)}\n\n")
        
        # Display technique-specific results
        self.results_text.insert(tk.END, f"TECHNIQUE DETAILS\n")
        self.results_text.insert(tk.END, f"{'='*30}\n")
        
        techniques = self.detection_results.get("techniques", {})
        for technique, results in techniques.items():
            self.results_text.insert(tk.END, f"{technique.capitalize()} Analysis:\n")
            for key, value in results.items():
                if key == "anomaly_detected":
                    status = "DETECTED" if value else "Not Detected"
                    self.results_text.insert(tk.END, f"  - Anomaly: {status}\n")
                elif key == "confidence":
                    self.results_text.insert(tk.END, f"  - Confidence: {value:.1f}%\n")
                else:
                    self.results_text.insert(tk.END, f"  - {key.replace('_', ' ').capitalize()}: {value}\n")
            self.results_text.insert(tk.END, "\n")
        
        # Display hidden data
        hidden_data = self.detection_results.get("hidden_data", "No hidden data detected")
        self.results_text.insert(tk.END, f"Hidden Data:\n{hidden_data}\n\n")
            
        # Create visualization of overall results
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
            
        # Risk gauge visualization
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Define gauge properties
        gauge_width = 0.3
        gauge_colors = {
            "Low": "green",
            "Medium": "orange",
            "High": "red"
        }
        
        confidence = self.detection_results["confidence_score"] / 100
        risk_color = gauge_colors[self.detection_results["overall_risk"]]
        
        # Create gauge
        ax.add_patch(plt.Circle((0.5, 0), 0.8, fill=False, color='gray'))
        ax.add_patch(patches.Wedge((0.5, 0), 0.8, 180, 180 + 180 * confidence, width=gauge_width, color=risk_color))  # Use patches.Wedge
        
        # Add labels
        ax.text(0.5, -0.1, f"{self.detection_results['confidence_score']:.1f}%", 
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.2, f"Risk Level: {self.detection_results['overall_risk']}", 
                ha='center', va='center', fontsize=10)
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 0.5)
        ax.axis('off')
        
        # Add the figure to the visualization frame
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plt.close(fig)
        
        # Create bar chart of technique confidences
        tech_names = []
        tech_scores = []
        tech_colors = []
        
        for technique, results in techniques.items():
            if "confidence" in results:
                tech_names.append(technique.capitalize())
                tech_scores.append(results["confidence"])
                tech_colors.append("red" if results.get("anomaly_detected", False) else "green")
        
        if tech_names:
            # Bar chart of technique confidences
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            bars = ax2.bar(tech_names, tech_scores, color=tech_colors)
            ax2.set_ylabel("Confidence (%)")
            ax2.set_title("Detection Confidence by Technique")
            
            # Add threshold line
            ax2.axhline(y=50, linestyle='--', color='gray')
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add the figure to the visualization frame
            canvas2 = FigureCanvasTkAgg(fig2, master=self.viz_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            plt.close(fig2)
    
    def extract_hidden_data(self, data, data_type):
        """Extract hidden data from the file"""
        hidden_data = "No hidden data detected"
        
        if data_type == "image" or data_type == "video":
            if isinstance(data, list):
                data = np.array(data)
            lsb_array = data % 2
            hidden_bits = lsb_array.flatten()
            hidden_bytes = np.packbits(hidden_bits)
            try:
                hidden_data = hidden_bytes.tobytes().decode('utf-8', errors='ignore')
                hidden_data = hidden_data.split('\x00', 1)[0]  # Stop at the first null character
            except UnicodeDecodeError:
                hidden_data = "Hidden data could not be decoded properly"
        elif data_type == "audio":
            hidden_bits = (data * 255).astype(np.uint8) % 2
            hidden_bytes = np.packbits(hidden_bits)
            try:
                hidden_data = hidden_bytes.tobytes().decode('utf-8', errors='ignore')
                hidden_data = hidden_data.split('\x00', 1)[0]  # Stop at the first null character
            except UnicodeDecodeError:
                hidden_data = "Hidden data could not be decoded properly"
        
        return hidden_data
    
    def save_results(self):
        """Save detection results to file"""
        if not self.detection_results:
            messagebox.showerror("Error", "No analysis results to save")
            return
            
        # Ask user for file location and format
        formats = [
            ("JSON files", "*.json"),
            ("CSV files", "*.csv"),
            ("Text files", "*.txt")
        ]
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=formats,
            initialfile=f"steganalysis_{os.path.basename(self.file_path)}"
        )
        
        if not file_path:
            return
            
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.json':
                with open(file_path, 'w') as f:
                    json.dump(self.detection_results, f, indent=4)
            elif ext == '.csv':
                # Flatten the nested structure for CSV
                flat_data = {
                    "file_name": self.detection_results["file_name"],
                    "file_type": self.detection_results["file_type"],
                    "analysis_date": self.detection_results["analysis_date"],
                    "overall_risk": self.detection_results["overall_risk"],
                    "confidence_score": self.detection_results["confidence_score"],
                    "anomaly_count": self.detection_results.get("anomaly_count", 0)
                }
                
                # Add technique-specific results
                techniques = self.detection_results.get("techniques", {})
                for technique, results in techniques.items():
                    for key, value in results.items():
                        flat_data[f"{technique}_{key}"] = value
                
                # Write to CSV
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=flat_data.keys())
                    writer.writeheader()
                    writer.writerow(flat_data)
            else:  # Text file
                with open(file_path, 'w') as f:
                    f.write(f"Steganography Detection Results\n")
                    f.write(f"{'='*40}\n\n")
                    
                    f.write(f"File: {self.detection_results['file_name']}\n")
                    f.write(f"Type: {self.detection_results['file_type']}\n")
                    f.write(f"Analysis Date: {self.detection_results['analysis_date']}\n\n")
                    
                    f.write(f"RESULTS SUMMARY\n")
                    f.write(f"{'-'*20}\n")
                    f.write(f"Overall Risk Level: {self.detection_results['overall_risk']}\n")
                    f.write(f"Confidence Score: {self.detection_results['confidence_score']:.1f}%\n")
                    f.write(f"Anomalies Detected: {self.detection_results.get('anomaly_count', 0)}\n\n")
                    
                    f.write(f"TECHNIQUE DETAILS\n")
                    f.write(f"{'-'*20}\n")
                    
                    techniques = self.detection_results.get("techniques", {})
                    for technique, results in techniques.items():
                        f.write(f"{technique.capitalize()} Analysis:\n")
                        for key, value in results.items():
                            if key == "anomaly_detected":
                                status = "DETECTED" if value else "Not Detected"
                                f.write(f"  - Anomaly: {status}\n")
                            elif key == "confidence":
                                f.write(f"  - Confidence: {value:.1f}%\n")
                            else:
                                f.write(f"  - {key.replace('_', ' ').capitalize()}: {value}\n")
                        f.write("\n")
            
            self.status_var.set(f"Results saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Results saved successfully to {file_path}")
            
        except Exception as e:
            self.status_var.set("Error saving results")
            messagebox.showerror("Error", f"Could not save results: {str(e)}")


# Main application file
def main():
    root = tk.Tk()
    app = SteganographyDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()