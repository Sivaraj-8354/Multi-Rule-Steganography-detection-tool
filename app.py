from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import wave
import io
from PIL import Image
import tempfile
import shutil

app = Flask(__name__, static_folder='static', static_url_path='')

# Create a directory for storing temporary files
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Cover text for text steganography
COVER_TEXT = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper, ligula eu tempor congue, eros est euismod turpis, id tincidunt sapien risus a quam. Maecenas fermentum consequat mi. Donec fermentum. Pellentesque malesuada nulla a mi. Duis sapien sem, aliquet nec, commodo eget, consequat quis, neque.""".split()

# Helper functions
def msgtobinary(msg):
    if type(msg) == str:
        return ''.join([format(ord(i), "08b") for i in msg])
    elif type(msg) == bytes or type(msg) == np.ndarray:
        return [format(i, "08b") for i in msg]
    elif type(msg) == int or type(msg) == np.uint8:
        return format(msg, "08b")
    else:
        raise TypeError("Input type is not supported")

# Image steganography functions
def encode_img_data(img, data):
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
        if index_data >= length_data:
            break
    
    return img

def decode_img_data(img):
    data_binary = ""
    for i in img:
        for pixel in i:
            r, g, b = msgtobinary(pixel)
            data_binary += r[-1]
            data_binary += g[-1]
            data_binary += b[-1]
            
            if len(data_binary) >= 24:  # Process in chunks for efficiency
                total_bytes = [data_binary[i: i+8] for i in range(0, len(data_binary), 8)]
                decoded_data = ""
                for byte in total_bytes:
                    if len(byte) == 8:  # Make sure we have a complete byte
                        decoded_data += chr(int(byte, 2))
                        if decoded_data[-5:] == "*^*^*":
                            return decoded_data[:-5]
                
                # Keep the last incomplete byte (if any)
                remainder = len(data_binary) % 8
                if remainder > 0:
                    data_binary = data_binary[-remainder:]
                else:
                    data_binary = ""
    
    # Process any remaining data
    if data_binary:
        total_bytes = [data_binary[i: i+8] for i in range(0, len(data_binary), 8)]
        decoded_data = ""
        for byte in total_bytes:
            if len(byte) == 8:
                decoded_data += chr(int(byte, 2))
                if decoded_data[-5:] == "*^*^*":
                    return decoded_data[:-5]
    
    return ""

# Text steganography functions
def encode_txt_data(text):
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
    
    result = []
    i = 0
    while i < len(res1):
        s = COVER_TEXT[int(i / 12)]
        j = 0
        x = ""
        HM_SK = ""
        while j < 12 and (i + j + 1) < len(res1):
            x = res1[j + i] + res1[i + j + 1]
            HM_SK += ZWC[x]
            j += 2
        s1 = s + HM_SK
        result.append(s1)
        i += 12
    
    t = int(len(res1) / 12)
    while t < len(COVER_TEXT):
        result.append(COVER_TEXT[t])
        t += 1
    
    return ' '.join(result)

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
        if a >= len(temp) or c >= len(temp):
            break
        
        t3 = temp[a:b]
        a += 12
        b += 12
        i += 12
        
        t4 = temp[c:d]
        c += 12
        d += 12
        
        if t3 == '0110':
            try:
                decimal_data = int(t4, 2)
                final += chr((decimal_data ^ 170) + 48)
            except:
                pass
        elif t3 == '0011':
            try:
                decimal_data = int(t4, 2)
                final += chr((decimal_data ^ 170) - 48)
            except:
                pass
    
    return final

# Audio steganography functions
def encode_aud_data(audio_path, data, output_path):
    song = wave.open(audio_path, mode='rb')
    nframes = song.getnframes()
    frames = song.readframes(nframes)
    frame_list = list(frames)
    frame_bytes = bytearray(frame_list)
    
    result = []
    for c in data:
        bits = bin(ord(c))[2:].zfill(8)
        result.extend([int(b) for b in bits])
    
    # Add terminator
    for c in '*^*^*':
        bits = bin(ord(c))[2:].zfill(8)
        result.extend([int(b) for b in bits])
    
    if len(result) > len(frame_bytes):
        raise ValueError("Audio file too small to encode this message")
    
    j = 0
    for i in range(len(result)):
        res = bin(frame_bytes[j])[2:].zfill(8)
        if len(res) >= 4:  # Make sure there's enough bits
            if (res[-4] == '1' and result[i] == 1) or (res[-4] == '0' and result[i] == 0):
                frame_bytes[j] = (frame_bytes[j] & 253)  # No change needed to LSB
            else:
                frame_bytes[j] = (frame_bytes[j] & 253) | 2  # Set the 2nd LSB
                frame_bytes[j] = (frame_bytes[j] & 254) | result[i]  # Set the LSB
        j += 1
    
    frame_modified = bytes(frame_bytes)
    
    with wave.open(output_path, 'wb') as fd:
        fd.setparams(song.getparams())
        fd.writeframes(frame_modified)
    
    song.close()

def decode_aud_data(file_path):
    extracted = ""
    terminator_found = False
    
    with wave.open(file_path, mode='rb') as song:
        nframes = song.getnframes()
        frames = song.readframes(nframes)
        frame_list = list(frames)
        frame_bytes = bytearray(frame_list)
        
        for i in range(len(frame_bytes)):
            res = bin(frame_bytes[i])[2:].zfill(8)
            if len(res) >= 4:  # Ensure we have enough bits
                if res[-2] == '0':  # Check 2nd LSB
                    extracted += res[-4]
                else:
                    extracted += res[-1]
            
            # Check for terminator every few bytes
            if len(extracted) >= 40 and len(extracted) % 8 == 0:
                all_bytes = [extracted[i:i+8] for i in range(0, len(extracted), 8)]
                decoded_data = ""
                for byte in all_bytes:
                    decoded_data += chr(int(byte, 2))
                    if decoded_data.endswith("*^*^*"):
                        return decoded_data[:-5]
    
    return ""

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/image/encode', methods=['POST'])
def encode_image():
    try:
        file = request.files['file']
        message = request.form['message']
        
        # Save the uploaded file
        temp_input = os.path.join(TEMP_DIR, 'input_' + file.filename)
        file.save(temp_input)
        
        # Read image, encode message, and save
        img = cv2.imread(temp_input)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        encoded_img = encode_img_data(img, message)
        temp_output = os.path.join(TEMP_DIR, 'output_' + file.filename)
        cv2.imwrite(temp_output, encoded_img)
        
        # Return encoded image
        return send_file(temp_output, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.remove(temp_input)
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.remove(temp_output)

@app.route('/api/image/decode', methods=['POST'])
def decode_image():
    try:
        file = request.files['file']
        
        # Save the uploaded file
        temp_input = os.path.join(TEMP_DIR, 'decode_' + file.filename)
        file.save(temp_input)
        
        # Read image and decode message
        img = cv2.imread(temp_input)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        message = decode_img_data(img)
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.remove(temp_input)

@app.route('/api/text/encode', methods=['POST'])
def api_text_encode():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        encoded_text = encode_txt_data(message)
        return jsonify({"encodedText": encoded_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/text/decode', methods=['POST'])
def api_text_decode():
    try:
        file = request.files['file']
        
        # Save the uploaded file
        temp_input = os.path.join(TEMP_DIR, 'decode_' + file.filename)
        file.save(temp_input)
        
        message = decode_txt_data(temp_input)
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.remove(temp_input)

@app.route('/api/audio/encode', methods=['POST'])
def api_audio_encode():
    try:
        file = request.files['file']
        message = request.form['message']
        
        # Save the uploaded file
        temp_input = os.path.join(TEMP_DIR, 'input_' + file.filename)
        file.save(temp_input)
        
        # Encode message and save
        temp_output = os.path.join(TEMP_DIR, 'output_' + file.filename)
        encode_aud_data(temp_input, message, temp_output)
        
        # Return encoded audio
        return send_file(temp_output, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.remove(temp_input)
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.remove(temp_output)

@app.route('/api/audio/decode', methods=['POST'])
def api_audio_decode():
    try:
        file = request.files['file']
        
        # Save the uploaded file
        temp_input = os.path.join(TEMP_DIR, 'decode_' + file.filename)
        file.save(temp_input)
        
        message = decode_aud_data(temp_input)
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.remove(temp_input)

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

# Clean up temp directory on exit
def cleanup():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Save index.html to static directory
    with open(os.path.join(static_dir, 'index.html'), 'w') as f:
        # You would normally paste the HTML content here
        # For this example, we'll use a placeholder
        f.write("""
        <!DOCTYPE html>
        <html>
        <head><title>Placeholder</title></head>
        <body><p>Please replace this file with your actual HTML.</p></body>
        </html>
        """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)