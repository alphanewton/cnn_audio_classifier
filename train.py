from pathlib import Path
import modal
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
from model import AudioCNN
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter


app = modal.App("audio-cnn")

image = (modal.Image.debian_slim() #docker linux container
         .pip_install_from_requirements("requirements.txt") #python dependencies
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"]) #system dependencies
         .run_commands([ #Sequence on commands on docker container
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc-50-data", create_if_missing=True) #Storing data
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True) #CNN trained Model file

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata["category"].unique())
        #numerize the class
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        #append to df
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]

        waveform, sample_rate  = torchaudio.load(audio_path) #.wav can be multiple channels [channels, samples]

        if waveform.shape[0] > 1: #[2, 44000] -> [2, 40000]
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform

        return spectogram, row["label"]

def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2) #Blending percentage
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :] #linear interpolation (0.7 * audio1) + (0.3 * audio2)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/models/tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir)


    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),
        T.FrequencyMasking(30), #Prevents overfitting
        T.TimeMasking(80) #Prevents overfitting
    )

    validation_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),    
    )

    train_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)

    val_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="test", transform=validation_transform)

    print("Training Samples:", len(train_dataset))
    print("Validation Samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=50).to(device)
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #Forces model to be more humble in predictions [1, 0, 0] -> [0.9, 0.03, 0.07]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.002, steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=0.1)

    best_accuracy = 0.0
    print("Starting Training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            #Data Mixing
            if np.random.random() > 0.7:
                data, target_a, target_b, lam= mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)


        #Validation after epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct/total
        avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)

        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f} Validation Loss: {avg_val_loss:.4f} Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "classes": train_dataset.classes,
            }, "/models/best_model.pth")
            print(f"Saved best model: {best_accuracy:.2f}%")
    
    writer.close()
    print(f"Training Complete! Best Accuracy: {best_accuracy:.2f}%")


@app.local_entrypoint()
def main():
    train.remote()
