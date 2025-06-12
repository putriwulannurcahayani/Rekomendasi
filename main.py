from flask import Flask, request, Response
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

app = Flask(__name__)

# Load model dan tokenizer
model = load_model('label_cacalan.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Parameter
max_len = 20
vocab_size = len(tokenizer.word_index) + 1

# Fungsi untuk generate review dari rating
def generate_review(seed_rating, max_len=20):
    result = []
    input_seq = [seed_rating]  # gunakan rating sebagai seed awal

    for _ in range(max_len):
        padded = pad_sequences([input_seq], maxlen=max_len, padding='post')
        prediction = model.predict(padded, verbose=0)
        predicted_id = np.argmax(prediction[0])

        if predicted_id == 0:
            break  # jika end token

        result.append(predicted_id)
        input_seq.append(predicted_id)

    # Ubah token ID ke kata
    reversed_word_index = {v: k for k, v in tokenizer.word_index.items()}
    predicted_words = [reversed_word_index.get(i, '') for i in result]
    return ' '.join(predicted_words).strip()

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'rating' not in data:
        return Response(json.dumps({"error": "Request JSON harus memiliki key 'rating'"}), mimetype='application/json', status=400)

    try:
        rating = int(data['rating'])
    except ValueError:
        return Response(json.dumps({"error": "Rating harus berupa angka 1 sampai 5"}), mimetype='application/json', status=400)

    if rating < 1 or rating > 5:
        return Response(json.dumps({"error": "Rating harus antara 1 sampai 5"}), mimetype='application/json', status=400)

    # Generate review dari rating
    generated_review = generate_review(seed_rating=rating, max_len=max_len)

    # Tambahkan dummy review jika hasil kosong
    if not generated_review:
        generated_review = "Pelayanan sangat bagus dan cepat!"

    kategori = "Buruk" if rating <= 2 else "Baik"

    # Kembalikan hasil dengan urutan sesuai permintaan
    response_data = {
        "review": generated_review,
        "rating": rating,
        "kategori": kategori
    }

    return Response(json.dumps(response_data), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
