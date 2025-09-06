import base64
import io
import modal 
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import librosa

from model import AudioCNN

app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["libsndfile1"])
         .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model")

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
            T.AmplitudeToDB(),
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0) #Add channel dimension
        spectogram = self.transform(waveform)
        return spectogram.unsqueeze(0) #Add batch dimension

class InferenceRequest(BaseModel):
    audio_data: str

@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=10)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading model on enter")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("/models/best_model.pth", map_location=self.device)
        self.classes = checkpoint["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")
    
    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        #frontend -> send file directly -> inference endpoint
        audio_bytes = base64.b64decode(request.audio_data)
        
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rate != 44100:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
        
        spectogram = self.audio_processor.process_audio_chunk(audio_data)
        spectogram = spectogram.to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(spectogram, return_feature_map=True)
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1) #dim=0 batch, dim=1 class(batch_size, num_classes)
            top3_probs, top3_indicies = torch.topk(probabilities[0], k=3)

            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()} for prob, idx in zip(top3_probs, top3_indicies)]

            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4: #[batch_size, channels, height, width]
                    aggregated_tensor = torch.mean(tensor, dim=1)
                    squeezed_tensor = aggregated_tensor.squeeze(0)
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": clean_array.shape,
                        "data": clean_array.tolist()
                    }

            spectogram_np = spectogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectogram = np.nan_to_num(spectogram_np)

            max_samples = 8000
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions": predictions, 
            "visualization": viz_data, 
            "input_spectogram": {
                "shape": list(clean_spectogram.shape),
                "data": clean_spectogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": 22050,
                "duration": len(waveform_data) / 22050
            }
        }
        return response


@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("1-100038-A-14.wav")
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json={"audio_data": audio_b64})
    response.raise_for_status()
    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values of waveform: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get('duration', 0)}")



        

        

