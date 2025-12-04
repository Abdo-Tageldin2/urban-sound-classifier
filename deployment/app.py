import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. Configuration & Classes ---
CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
    'siren', 'street_music'
]

# -- ResNet Architecture --
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# -- ANN Architecture --
class AudioANN(nn.Module):
    def __init__(self):
        super(AudioANN, self).__init__()
        self.layer1 = nn.Linear(40, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        # Note: Dropout is disabled automatically in model.eval() mode
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.output(x)
        return x

# --- 2. Loading Functions ---

@st.cache_resource
def load_all_models():
    models = {}
    
    # 1. Load Scikit-Learn Components (SVM, PCA, Scaler)
    try:
        models['svm'] = joblib.load('model_svm.joblib')
        models['pca'] = joblib.load('pca_transformer.joblib')
        models['scaler'] = joblib.load('scaler.joblib') # <--- NEW: Load Scaler
    except:
        pass 
        
    # 2. ANN
    try:
        ann = AudioANN()
        ann.load_state_dict(torch.load('best_audio_ann.pth', map_location='cpu'))
        ann.eval()
        models['ann'] = ann
    except:
        pass
        
    # 3. ResNet
    try:
        resnet = AudioResNet(num_classes=10)
        resnet.load_state_dict(torch.load('best_audio_resnet3.pth', map_location='cpu'))
        resnet.eval()
        models['resnet'] = resnet
    except:
        pass
        
    return models

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    
    # A. MFCCs (For SVM/ANN)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # B. Spectrogram (For ResNet)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize/Pad to 128x128
    target_size = 128
    if mel_spec_db.shape[1] < target_size:
        pad = target_size - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0,pad)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :target_size]
    
    return mfccs_scaled, mel_spec_db

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Urban Sound Ensemble", page_icon="🎧", layout="wide")

st.title("🏙️ Urban Sound Ensemble Classifier")
st.markdown("Upload an audio file to see how the **SVM**, **ANN**, and **ResNet** models classify it individually and vote.")

uploaded_file = st.file_uploader("Upload Audio (WAV, MP3)", type=['wav', 'mp3', 'ogg'])

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Analyze Audio"):
        with st.spinner("Processing..."):
            models = load_all_models()
            
            if not models:
                st.error("No models loaded! Ensure .pth and .joblib files exist.")
            else:
                # Extract Features
                mfcc_feat, spec_feat = extract_features(uploaded_file)
                raw_mfcc_input = mfcc_feat.reshape(1, -1)
                
                # --- PRE-PROCESSING PIPELINE ---
                # 1. Scale Data (Crucial for SVM and ANN)
                if 'scaler' in models:
                    scaled_mfcc_input = models['scaler'].transform(raw_mfcc_input)
                else:
                    st.warning("Scaler missing. Predictions may be inaccurate.")
                    scaled_mfcc_input = raw_mfcc_input 

                # 2. Prepare Inputs
                ann_input = torch.tensor(scaled_mfcc_input).float()
                resnet_input = torch.tensor(spec_feat).float().unsqueeze(0).unsqueeze(0)
                
                results = {}
                votes = []

                # --- MODEL 1: SVM ---
                svm_res = {"pred": "Error", "conf": 0.0}
                if 'svm' in models and 'pca' in models:
                    try:
                        # Pipeline: Scaled Input -> PCA -> SVM
                        svm_input_pca = models['pca'].transform(scaled_mfcc_input)
                        pred_idx = models['svm'].predict(svm_input_pca)[0]
                        svm_res["pred"] = CLASSES[pred_idx]
                        votes.append(svm_res["pred"])
                        
                        if hasattr(models['svm'], "predict_proba"):
                            probs = models['svm'].predict_proba(svm_input_pca)
                            svm_res["conf"] = np.max(probs)
                        else:
                            svm_res["conf"] = None
                    except Exception as e:
                        svm_res["pred"] = f"Err: {e}"
                results['SVM'] = svm_res

                # --- MODEL 2: ANN ---
                ann_res = {"pred": "Error", "conf": 0.0}
                if 'ann' in models:
                    with torch.no_grad():
                        # Pipeline: Scaled Input -> ANN
                        outputs = models['ann'](ann_input)
                        probs = F.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)
                        ann_res["pred"] = CLASSES[pred.item()]
                        ann_res["conf"] = conf.item()
                        votes.append(ann_res["pred"])
                results['ANN'] = ann_res

                # --- MODEL 3: ResNet ---
                resnet_res = {"pred": "Error", "conf": 0.0}
                if 'resnet' in models:
                    with torch.no_grad():
                        outputs = models['resnet'](resnet_input)
                        probs = F.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)
                        resnet_res["pred"] = CLASSES[pred.item()]
                        resnet_res["conf"] = conf.item()
                        votes.append(resnet_res["pred"])
                results['ResNet'] = resnet_res

                # --- VISUALIZE RESULTS ---
                st.write("### 🧠 Individual Model Analysis")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.info("🤖 **SVM** (PCA+Scaled)")
                    st.write(f"**Prediction:** {results['SVM']['pred'].replace('_', ' ').title()}")
                    if results['SVM']['conf'] is not None:
                        st.progress(float(results['SVM']['conf']))

                with col2:
                    st.warning("🧠 **ANN** (Scaled)")
                    st.write(f"**Prediction:** {results['ANN']['pred'].replace('_', ' ').title()}")
                    st.progress(float(results['ANN']['conf']))

                with col3:
                    st.success("👁️ **ResNet** (Spatial)")
                    st.write(f"**Prediction:** {results['ResNet']['pred'].replace('_', ' ').title()}")
                    st.progress(float(results['ResNet']['conf']))

                # --- FINAL ENSEMBLE VOTE ---
                if votes:
                    vote_counts = Counter(votes)
                    winner, count = vote_counts.most_common(1)[0]
                    
                    # Tie-breaker logic: Prefer ResNet
                    if count == 1 and 'ResNet' in results:
                        winner = results['ResNet']['pred']
                        tie_text = "(Tie-breaker: ResNet)"
                    else:
                        tie_text = ""

                    st.divider()
                    st.markdown(f"<h2 style='text-align: center; color: green;'>🏆 Consensus: {winner.upper().replace('_', ' ')}</h2>", unsafe_allow_html=True)
                    if tie_text: st.caption(tie_text)
                
                # --- SPECTROGRAM ---
                st.write("---")
                st.write("**ResNet View (Spectrogram):**")
                fig, ax = plt.subplots(figsize=(10, 3))
                img = librosa.display.specshow(spec_feat, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                st.pyplot(fig)