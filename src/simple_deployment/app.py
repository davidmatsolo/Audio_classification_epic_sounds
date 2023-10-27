from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib

import librosa
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('D:/big data/practice/big_data_project/src/models/model_20231026111356.h5')
#corresponding encoder
loaded_label_encoder = joblib.load('D:/big data/practice/big_data_project/src/label_encoders/label_encoder_20231026111356.pkl')
    
@app.route('/', methods=['GET'])
def hello_word():
    #print()
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    audiofile= request.files['audiofile']
    audio_path = "D:/big data/practice/big_data_project/sounds/" + audiofile.filename
    audiofile.save(audio_path)

    
    print("audio path ", audio_path)
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast') 
    mel_spectrogram_features = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    log_mel_spectrogram_scaled_features = np.mean(mel_spectrogram_features.T,axis=0)
    print("mfccs_scaled_features ", log_mel_spectrogram_scaled_features)
    print
    log_mel_spectrogram_scaled_features=log_mel_spectrogram_scaled_features.reshape(1,-1)
    log_mel_spectrogram_scaled_features = log_mel_spectrogram_scaled_features[:, :35]
    print("new mfccs_scaled_features  ", log_mel_spectrogram_scaled_features)
    print("Shape ", log_mel_spectrogram_scaled_features.shape)
    print
    

    predictions=model.predict(log_mel_spectrogram_scaled_features)
    print("predictions ", predictions)
    predicted_class_index = np.argmax(predictions)
    #encoded_labels = loaded_label_encoder.transform(mfccs_scaled_features)
    predicted_class = loaded_label_encoder.inverse_transform([predicted_class_index])
    print
    print("predicted_class ", predicted_class)

    classification = predicted_class[0]
    
    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)


