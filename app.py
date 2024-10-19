from flask import Flask, request, jsonify
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import torchaudio
import os

app = Flask(__name__)

# Path to your locally saved model files
MODEL_DIR = "/path/to/local_model_directory"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  # Pretrained tokenizer
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, from_safetensors=True)

# Load model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define allowed extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_wav_file(filepath):
    try:
        waveform, sr = torchaudio.load(filepath)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values[0]

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Validate file
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join("/tmp", file.filename)  # Temporary file path for processing
        file.save(filepath)

        # Process the wav file
        input_values = process_wav_file(filepath)
        if input_values is None:
            return jsonify({"error": "Error processing audio file"}), 500

        input_values = input_values.unsqueeze(0).to(device)

        # Make predictions
        model.eval()
        with torch.no_grad():
            logits = model(input_values).logits

        preds = torch.sigmoid(logits).cpu().numpy()

        # Binary classification: decide based on threshold
        prediction = (preds > 0.5).astype(int)

        return jsonify({"prediction": prediction.tolist()})

    else:
        return jsonify({"error": "Invalid file type. Please upload a .wav file."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
