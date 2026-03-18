from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os

app = FastAPI(title="MIR Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav"}
MAX_FILE_SIZE_MB = 25

def estimate_sections(total_bars: int):
    if total_bars <= 0:
        return []

    template = [
        ("Intro", 8),
        ("Verse", 16),
        ("Chorus", 16),
        ("Verse", 16),
        ("Chorus", 16),
        ("Bridge", 8),
        ("Chorus", 16),
        ("Outro", 8),
    ]

    sections = []
    remaining = total_bars

    for name, bars in template:
        if remaining <= 0:
            break
        current = min(bars, remaining)
        sections.append({"name": name, "bars": int(current)})
        remaining -= current

    return sections

def analyze_song(file_path: str):
    y, sr = librosa.load(file_path, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    total_beats = len(beats)
    beats_per_bar = 4
    total_bars = int(total_beats / beats_per_bar)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_index = int(np.argmax(chroma_mean))
    estimated_key = pitch_classes[key_index]

    rms = librosa.feature.rms(y=y)[0]
    energy_curve = rms.tolist()

    sections = estimate_sections(total_bars)

    return {
        "bpm": round(float(tempo), 2),
        "time_signature": "4/4",
        "total_bars": total_bars,
        "key": estimated_key,
        "sections": sections,
        "energy_curve": energy_curve,
        "chords": []
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "MIR Analyzer running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato no permitido. Usa WAV o MP3.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail="Archivo demasiado grande.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
        temp.write(contents)
        temp_path = temp.name

    try:
        analysis = analyze_song(temp_path)
        return analysis
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
