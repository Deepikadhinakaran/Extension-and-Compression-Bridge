"""
============================================================
  AI SOUND RECOGNITION MODULE
  Train Horn Detection using Machine Learning
  Expansion & Compression Bridge Project
============================================================
  This module:
  1. Records audio samples
  2. Extracts MFCC features (Mel Frequency Cepstral Coefficients)
  3. Trains a simple classifier to distinguish train sounds
  4. Runs real-time inference on the bridge system
============================================================
  Install dependencies:
    pip install numpy scikit-learn librosa sounddevice joblib
============================================================
"""

import numpy as np
import os
import joblib
import sounddevice as sd
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000   # Hz
DURATION        = 1.0     # seconds per sample
N_MFCC          = 13      # MFCC feature count
MODEL_PATH      = "train_sound_model.pkl"
SCALER_PATH     = "train_sound_scaler.pkl"


# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────
def extract_features(audio, sr=SAMPLE_RATE):
    """
    Extracts MFCC + additional features from audio sample.
    Returns a 1D feature vector.
    """
    try:
        import librosa

        # MFCCs - captures timbral texture
        mfcc = librosa.feature.mfcc(y=audio.astype(float),
                                     sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)  # shape: (13,)
        mfcc_std  = np.std(mfcc,  axis=1)  # shape: (13,)

        # Spectral Centroid - brightness of sound
        centroid = librosa.feature.spectral_centroid(y=audio.astype(float), sr=sr)
        centroid_mean = np.mean(centroid)

        # Zero Crossing Rate - noisiness
        zcr = librosa.feature.zero_crossing_rate(audio.astype(float))
        zcr_mean = np.mean(zcr)

        # RMS Energy - loudness
        rms = librosa.feature.rms(y=audio.astype(float))
        rms_mean = np.mean(rms)

        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [centroid_mean, zcr_mean, rms_mean]
        ])
        return features

    except ImportError:
        # Fallback: FFT-based features if librosa unavailable
        fft_mag = np.abs(np.fft.rfft(audio))
        freq    = np.fft.rfftfreq(len(audio), d=1.0 / sr)
        # Select features in key frequency bands
        bands = [(0, 200), (200, 800), (800, 2000), (2000, 4000), (4000, 8000)]
        feat  = []
        for lo, hi in bands:
            mask = (freq >= lo) & (freq < hi)
            feat.append(np.mean(fft_mag[mask]) if np.any(mask) else 0)
        return np.array(feat)


# ─────────────────────────────────────────────────────────
# AUDIO RECORDING
# ─────────────────────────────────────────────────────────
def record_sample(duration=DURATION, sr=SAMPLE_RATE):
    """Records audio from microphone. Returns numpy array."""
    print(f"[AUDIO] Recording for {duration}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def load_wav(filepath, sr=SAMPLE_RATE):
    """Loads a WAV file as numpy array."""
    rate, data = wavfile.read(filepath)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data


# ─────────────────────────────────────────────────────────
# DATASET PREPARATION
# (Assumes labeled WAV files in folders: train_sounds/, other_sounds/)
# ─────────────────────────────────────────────────────────
def prepare_dataset(train_dir="train_sounds", other_dir="other_sounds"):
    """
    Loads labeled audio files and extracts features.
    Label 1 = train sound, Label 0 = ambient/other sound.
    """
    X, y = [], []

    def load_folder(folder, label):
        if not os.path.exists(folder):
            print(f"[DATASET] Folder '{folder}' not found. Skipping.")
            return
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                path  = os.path.join(folder, fname)
                audio = load_wav(path)
                feats = extract_features(audio)
                X.append(feats)
                y.append(label)
                print(f"[DATASET] Loaded: {fname} -> Label {label}")

    load_folder(train_dir, label=1)
    load_folder(other_dir, label=0)

    if not X:
        # Generate synthetic demo data for testing
        print("[DATASET] No files found. Using synthetic data for demo.")
        np.random.seed(42)
        X = np.random.rand(100, N_MFCC * 2 + 3)
        y = [1 if i < 50 else 0 for i in range(100)]
        return np.array(X), np.array(y)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────
def train_model():
    """Trains and saves the sound classifier."""
    print("\n[TRAIN] Preparing dataset...")
    X, y = prepare_dataset()

    print(f"[TRAIN] Dataset: {len(X)} samples | Classes: {np.bincount(y.astype(int))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n[TRAIN] Evaluation Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Other Sound", "Train Sound"]))

    # Save model & scaler
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[TRAIN] Model saved to '{MODEL_PATH}'")
    print(f"[TRAIN] Scaler saved to '{SCALER_PATH}'")


# ─────────────────────────────────────────────────────────
# REAL-TIME INFERENCE
# ─────────────────────────────────────────────────────────
class TrainSoundDetector:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model  = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("[AI] Model loaded successfully.")
        else:
            self.model  = None
            self.scaler = None
            print("[AI] No model found. Run train_model() first.")

    def is_train_sound(self, audio=None):
        """
        Detects whether the current audio is a train sound.
        If audio=None, records a new sample.
        Returns True if train sound detected.
        """
        if self.model is None:
            # Fallback to frequency analysis
            return self._frequency_fallback(audio)

        if audio is None:
            audio = record_sample()

        features = extract_features(audio)
        features = self.scaler.transform([features])
        prediction = self.model.predict(features)[0]
        confidence = max(self.model.predict_proba(features)[0])

        label = "TRAIN SOUND" if prediction == 1 else "Other Sound"
        print(f"[AI] Prediction: {label} | Confidence: {confidence:.2%}")

        return prediction == 1

    def _frequency_fallback(self, audio):
        """FFT-based fallback if no ML model is available."""
        if audio is None:
            audio = record_sample()
        fft_freq    = np.fft.rfftfreq(len(audio), d=1.0 / SAMPLE_RATE)
        fft_mag     = np.abs(np.fft.rfft(audio))
        dominant_f  = fft_freq[np.argmax(fft_mag)]
        print(f"[AI-Fallback] Dominant Freq: {dominant_f:.1f} Hz")
        return 200 <= dominant_f <= 800


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("=== TRAINING AI SOUND CLASSIFIER ===")
        train_model()
    else:
        print("=== REAL-TIME TRAIN SOUND DETECTION ===")
        detector = TrainSoundDetector()
        print("Listening... Press Ctrl+C to stop.\n")
        try:
            while True:
                detected = detector.is_train_sound()
                if detected:
                    print("⚠️  TRAIN DETECTED! Compress bridge!")
                else:
                    print("✅  No train. Bridge safe to extend.")
                import time; time.sleep(1)
        except KeyboardInterrupt:
            print("\n[AI] Detection stopped.")
