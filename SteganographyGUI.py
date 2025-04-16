import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
import wave
from PIL import Image, ImageTk

class SteganographyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Tool")
        self.root.geometry("600x400")
        self.root.configure(bg="#2c3e50")
        
        # Styling
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 14), background="#2c3e50", foreground="#ecf0f1")
        self.style.configure("TEntry", font=("Helvetica", 12))
        
        # Main Menu
        self.create_main_menu()
    
    def create_main_menu(self):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Title
        title_label = ttk.Label(self.root, text="Steganography Tool", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=20)
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg="#2c3e50")
        btn_frame.pack(pady=20)
        
        buttons = [
            ("Image Steganography", self.open_image_window),
            ("Text Steganography", self.open_text_window),
            ("Audio Steganography", self.open_audio_window),
            ("Exit", self.root.quit)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(btn_frame, text=text, command=command, width=20)
            btn.pack(pady=10)
            
    def create_operation_window(self, title, encode_func, decode_func, is_encode=True):
        # Create new window for operation
        op_window = tk.Toplevel(self.root)
        op_window.title(title)
        op_window.geometry("500x600")
        op_window.configure(bg="#34495e")

        # Title
        ttk.Label(op_window, text=title, font=("Helvetica", 20, "bold")).pack(pady=20)

        # File Selection
        file_frame = tk.Frame(op_window, bg="#34495e")
        file_frame.pack(pady=10, fill="x", padx=20)

        ttk.Label(file_frame, text="Select File:").pack(side="left")
        file_entry = ttk.Entry(file_frame)
        file_entry.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file(file_entry, title)).pack(side="left")

        # Message Input (for encoding)
        if is_encode:
            ttk.Label(op_window, text="Message:").pack(pady=5)
            msg_text = tk.Text(op_window, height=4, font=("Helvetica", 12))
            msg_text.pack(pady=5, padx=20, fill="x")
        else:
            msg_text = None

        # Output File (for encoding)
        if is_encode:
            output_frame = tk.Frame(op_window, bg="#34495e")
            output_frame.pack(pady=10, fill="x", padx=20)

            ttk.Label(output_frame, text="Output File:").pack(side="left")
            output_entry = ttk.Entry(output_frame)
            output_entry.pack(side="left", fill="x", expand=True, padx=5)
            ttk.Button(output_frame, text="Browse", command=lambda: self.browse_output_file(output_entry, title)).pack(side="left")
        else:
            output_entry = None

        # Status
        status_label = ttk.Label(op_window, text="", font=("Helvetica", 10))
        status_label.pack(pady=10)

        # Execute Button
        ttk.Button(op_window, text="Execute", command=lambda: self.execute_operation(
            encode_func if is_encode else decode_func, file_entry, msg_text, output_entry, status_label, is_encode
        )).pack(pady=20)

        # Back Button
        ttk.Button(op_window, text="Back", command=op_window.destroy).pack(pady=10)

    def browse_file(self, entry, title):
        # Normalize title to match keys in filetypes dictionary
        normalized_title = title.split(" - ")[0]  # Extract base title (e.g., "Image Steganography")
        filetypes = {
            "Image Steganography": [("Image files", "*.jpg *.png")],
            "Text Steganography": [("Text files", "*.txt")],
            "Audio Steganography": [("Audio files", "*.wav")]
        }
        file = filedialog.askopenfilename(filetypes=filetypes[normalized_title])
        if file:
            entry.delete(0, tk.END)
            entry.insert(0, file)
            
    def browse_output_file(self, entry, title):
        filetypes = {
            "Image Steganography": [("Image files", "*.jpg *.png")],
            "Text Steganography": [("Text files", "*.txt")],
            "Audio Steganography": [("Audio files", "*.wav")]
        }
        file = filedialog.asksaveasfilename(filetypes=filetypes[title], defaultextension=filetypes[title][0][1])
        if file:
            entry.delete(0, tk.END)
            entry.insert(0, file)
            
    def execute_operation(self, func, file_entry, msg_text, output_entry, status_label, is_encode):
        try:
            file_path = file_entry.get()
            output_path = output_entry.get() if output_entry else ""
            message = msg_text.get("1.0", tk.END).strip() if msg_text else ""
            
            if not file_path:
                raise ValueError("Please select an input file.")
            if is_encode and not output_path:
                raise ValueError("Please select an output file.")
            if is_encode and not message:
                raise ValueError("Please enter a message to encode.")
            
            status_label.config(text="Processing...", foreground="#f1c40f")
            
            if func == encode_img_data or func == decode_img_data:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Invalid image file.")
                if is_encode:
                    encode_img_data(img, message, output_path)
                    status_label.config(text="Encoded successfully!", foreground="#2ecc71")
                else:
                    decoded = decode_img_data(img)
                    if decoded:
                        if msg_text:
                            msg_text.delete("1.0", tk.END)
                            msg_text.insert("1.0", decoded)
                        messagebox.showinfo("Decoded Message", f"The hidden message is:\n{decoded}")
                        status_label.config(text="Decoded successfully!", foreground="#2ecc71")
                    else:
                        raise ValueError("No hidden message found.")
                        
            elif func == encode_txt_data or func == decode_txt_data:
                if is_encode:
                    encode_txt_data(message, output_path)
                    status_label.config(text="Encoded successfully!", foreground="#2ecc71")
                else:
                    decoded = decode_txt_data(file_path)
                    if decoded:
                        if msg_text:
                            msg_text.delete("1.0", tk.END)
                            msg_text.insert("1.0", decoded)
                        messagebox.showinfo("Decoded Message", f"The hidden message is:\n{decoded}")
                        status_label.config(text="Decoded successfully!", foreground="#2ecc71")
                    else:
                        raise ValueError("No hidden message found.")
                        
            elif func == encode_aud_data or func == decode_aud_data:
                if is_encode:
                    encode_aud_data(file_path, message, output_path)
                    status_label.config(text="Encoded successfully!", foreground="#2ecc71")
                else:
                    decoded = decode_aud_data(file_path)
                    if decoded:
                        if msg_text:
                            msg_text.delete("1.0", tk.END)
                            msg_text.insert("1.0", decoded)
                        messagebox.showinfo("Decoded Message", f"The hidden message is:\n{decoded}")
                        status_label.config(text="Decoded successfully!", foreground="#2ecc71")
                    else:
                        raise ValueError("No hidden message found.")
                        
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}", foreground="#e74c3c")
            
    def open_image_window(self):
        self.create_encode_decode_window("Image Steganography", encode_img_data, decode_img_data)

    def open_text_window(self):
        self.create_encode_decode_window("Text Steganography", encode_txt_data, decode_txt_data)

    def open_audio_window(self):
        self.create_encode_decode_window("Audio Steganography", encode_aud_data, decode_aud_data)

    def create_encode_decode_window(self, title, encode_func, decode_func):
        # Create new window
        ed_window = tk.Toplevel(self.root)
        ed_window.title(title)
        ed_window.geometry("400x300")
        ed_window.configure(bg="#34495e")

        # Title
        ttk.Label(ed_window, text=title, font=("Helvetica", 20, "bold")).pack(pady=20)

        # Buttons for Encode and Decode
        btn_frame = tk.Frame(ed_window, bg="#34495e")
        btn_frame.pack(pady=50)

        ttk.Button(btn_frame, text="Encode", command=lambda: self.create_operation_window(
            f"{title} - Encode", encode_func, decode_func, is_encode=True
        )).pack(side="left", padx=20)

        ttk.Button(btn_frame, text="Decode", command=lambda: self.create_operation_window(
            f"{title} - Decode", encode_func, decode_func, is_encode=False
        )).pack(side="left", padx=20)

        # Back Button
        ttk.Button(ed_window, text="Back", command=ed_window.destroy).pack(pady=20)

if __name__ == "__main__":
    # Define required functions
    def msgtobinary(msg):
        if type(msg) == str:
            return ''.join([format(ord(i), "08b") for i in msg])
        elif type(msg) == bytes or type(msg) == np.ndarray:
            return [format(i, "08b") for i in msg]
        elif type(msg) == int or type(msg) == np.uint8:
            return format(msg, "08b")
        else:
            raise TypeError("Input type is not supported")

    def encode_txt_data(text, output_file):
        l = len(text)
        i = 0
        add = ''
        while i < l:
            t = ord(text[i])
            if 32 <= t <= 64:
                t1 = t + 48
                t2 = t1 ^ 170
                res = bin(t2)[2:].zfill(8)
                add += "0011" + res
            else:
                t1 = t - 48
                t2 = t1 ^ 170
                res = bin(t2)[2:].zfill(8)
                add += "0110" + res
            i += 1
        res1 = add + "111111111111"
        HM_SK = ""
        ZWC = {"00": '\u200C', "01": '\u202C', "11": '\u202D', "10": '\u200E'}
        with open("Sample_cover_files/cover_text.txt", "r") as file1:
            with open(output_file, "w+", encoding="utf-8") as file3:
                word = []
                for line in file1:
                    word += line.split()
                i = 0
                while i < len(res1):
                    s = word[int(i / 12)]
                    j = 0
                    x = ""
                    HM_SK = ""
                    while j < 12:
                        x = res1[j + i] + res1[i + j + 1]
                        HM_SK += ZWC[x]
                        j += 2
                    s1 = s + HM_SK
                    file3.write(s1)
                    file3.write(" ")
                    i += 12
                t = int(len(res1) / 12)
                while t < len(word):
                    file3.write(word[t])
                    file3.write(" ")
                    t += 1

    def decode_txt_data(file_path):
        ZWC_reverse = {'\u200C': "00", '\u202C': "01", '\u202D': "11", '\u200E': "10"}
        temp = ''
        with open(file_path, "r", encoding="utf-8") as file4:
            for line in file4:
                for words in line.split():
                    T1 = words
                    binary_extract = ""
                    for letter in T1:
                        if letter in ZWC_reverse:
                            binary_extract += ZWC_reverse[letter]
                    if binary_extract == "111111111111":
                        break
                    else:
                        temp += binary_extract
        i = 0
        a = 0
        b = 4
        c = 4
        d = 12
        final = ''
        while i < len(temp):
            t3 = temp[a:b]
            a += 12
            b += 12
            i += 12
            t4 = temp[c:d]
            c += 12
            d += 12
            if t3 == '0110':
                decimal_data = int(t4, 2)
                final += chr((decimal_data ^ 170) + 48)
            elif t3 == '0011':
                decimal_data = int(t4, 2)
                final += chr((decimal_data ^ 170) - 48)
        return final

    def encode_img_data(img, data, nameoffile):
        if len(data) == 0:
            raise ValueError('Data entered to be encoded is empty')
        no_of_bytes = (img.shape[0] * img.shape[1] * 3) // 8
        if len(data) > no_of_bytes:
            raise ValueError("Insufficient bytes, need bigger image or less data!")
        data += '*^*^*'
        binary_data = msgtobinary(data)
        length_data = len(binary_data)
        index_data = 0
        for i in img:
            for pixel in i:
                r, g, b = msgtobinary(pixel)
                if index_data < length_data:
                    pixel[0] = int(r[:-1] + binary_data[index_data], 2)
                    index_data += 1
                if index_data < length_data:
                    pixel[1] = int(g[:-1] + binary_data[index_data], 2)
                    index_data += 1
                if index_data < length_data:
                    pixel[2] = int(b[:-1] + binary_data[index_data], 2)
                    index_data += 1
                if index_data >= length_data:
                    break
        cv2.imwrite(nameoffile, img)

    def decode_img_data(img):
        data_binary = ""
        for i in img:
            for pixel in i:
                r, g, b = msgtobinary(pixel)
                data_binary += r[-1]
                data_binary += g[-1]
                data_binary += b[-1]
                total_bytes = [data_binary[i: i+8] for i in range(0, len(data_binary), 8)]
                decoded_data = ""
                for byte in total_bytes:
                    decoded_data += chr(int(byte, 2))
                    if decoded_data[-5:] == "*^*^*":
                        return decoded_data[:-5]
        return ""

    def encode_aud_data(nameoffile, data, stegofile):
        song = wave.open(nameoffile, mode='rb')
        nframes = song.getnframes()
        frames = song.readframes(nframes)
        frame_list = list(frames)
        frame_bytes = bytearray(frame_list)
        res = ''.join(format(i, '08b') for i in bytearray(data, encoding='utf-8'))
        data += '*^*^*'
        result = []
        for c in data:
            bits = bin(ord(c))[2:].zfill(8)
            result.extend([int(b) for b in bits])
        j = 0
        for i in range(0, len(result), 1):
            res = bin(frame_bytes[j])[2:].zfill(8)
            if res[len(res)-4] == result[i]:
                frame_bytes[j] = (frame_bytes[j] & 253)
            else:
                frame_bytes[j] = (frame_bytes[j] & 253) | 2
                frame_bytes[j] = (frame_bytes[j] & 254) | result[i]
            j += 1
        frame_modified = bytes(frame_bytes)
        with wave.open(stegofile, 'wb') as fd:
            fd.setparams(song.getparams())
            fd.writeframes(frame_modified)
        song.close()

    def decode_aud_data(file_path):
        extracted = ""
        p = 0
        with wave.open(file_path, mode='rb') as song:
            nframes = song.getnframes()
            frames = song.readframes(nframes)
            frame_list = list(frames)
            frame_bytes = bytearray(frame_list)

            for i in range(len(frame_bytes)):
                if p == 1:
                    break
                res = bin(frame_bytes[i])[2:].zfill(8)
                if res[len(res)-2] == '0':
                    extracted += res[len(res)-4]
                else:
                    extracted += res[len(res)-1]
            
            all_bytes = [extracted[i: i+8] for i in range(0, len(extracted), 8)]
            decoded_data = ""
            for byte in all_bytes:
                decoded_data += chr(int(byte, 2))
                if decoded_data[-5:] == "*^*^*":
                    return decoded_data[:-5]
        return "No data found."

    root = tk.Tk()
    app = SteganographyGUI(root)
    root.mainloop()