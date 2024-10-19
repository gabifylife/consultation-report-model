from flask import Flask, request, jsonify
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio

app = Flask(__name__)


MODEL_PATH = 'model.safetensors'
CONFIG_PATH = 'config.json'

processor = Wav2Vec2Processor.from_pretrained(CONFIG_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(CONFIG_PATH)
model.load_state_dict(torch.load(MODEL_PATH))

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    waveform, sr = torchaudio.load(audio_file)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Assume binary classification for this example
    preds = torch.sigmoid(logits) > 0.5

    return jsonify({'prediction': preds.cpu().numpy().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
