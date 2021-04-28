from flask import Flask,request,jsonify
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from pydub import AudioSegment

app = Flask(__name__)

loaded_model = pickle.load(open('KNN_classifier.model', 'rb'))

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def prediction_emotion(model, audio_file):
    sound = AudioSegment.from_wav(audio_file)
    sound = sound.set_channels(1)
    sound.export("./audio.wav", format="wav")
    feature = extract_feature('./audio.wav', mfcc=True, chroma=True, mel=True)
    prediction = model.predict(feature.reshape(1, -1))
    return str(prediction[0])

@app.route('/api/speech_prediction', methods=['POST'])
def speech_prediction_api():
    results = prediction_emotion(loaded_model,request.files['speech'])
    return results

@app.route('/api/test')
def test_api():
    return 'Hello World'



if __name__ == '__main__':
    app.run()