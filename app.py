from flask import Flask, request, jsonify
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np

app = Flask(__name__)

MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB

CATEGORY_SUGGESTIONS = {
    "buffalo": "Buffalo adalah ...",
    "cheetah": "Cheetah adalah ...",
    "elephant": "Elephant adalah ...",
    "lion": "Lion adalah ...",
    "tiger": "Tiger adalah ...",
    "zebra": "Zebra adalah ...",
    # Tambahkan kategori lain di sini
}


# Fungsi untuk memuat model dari URL publik
def load_model_from_local():
    try:
        # Ganti 'path/to/your/model.h5' dengan path lokal dari file model
        model_path = "new_model.h5"

        # Muat model dari direktori lokal
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")

        return model
    except Exception as e:
        print(f"Error during model loading: {e}")
        return None


# Muat model sekali saat aplikasi mulai
model = load_model_from_local()


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"status": "fail", "message": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"status": "fail", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        if request.content_length > MAX_CONTENT_LENGTH:
            return jsonify(
                {
                    "status": "fail",
                    "message": "Payload content length greater than maximum allowed: 1000000",
                }
            ), 413

        secure_filename(file.filename)
        try:
            if model is None:
                raise Exception("Model not loaded")
            print("Predicting the image")
            result = model_predict(file)
            print(f"Prediction result: {result}")
            response_id = str(uuid.uuid4())
            suggestion = CATEGORY_SUGGESTIONS.get(
                result,
            )
            response_data = {
                "id": response_id,
                "result": result,
                "suggestion": suggestion,
                "createdAt": datetime.utcnow().isoformat() + "Z",
            }
            return jsonify(
                {
                    "status": "success",
                    "message": "Model is predicted successfully",
                    "data": response_data,
                }
            ), 201
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify(
                {
                    "status": "fail",
                    "message": "Terjadi kesalahan dalam melakukan prediksi",
                }
            ), 400
    else:
        return jsonify({"status": "fail", "message": "Invalid file format"}), 400


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_predict(file):
    try:
        # Proses file menjadi array yang bisa diterima model
        img = tf.io.decode_image(file.read(), channels=3)
        img = tf.image.resize(img, [224, 224])  # Sesuaikan ukuran sesuai model Anda
        img = tf.expand_dims(img, 0)  # Tambahkan dimensi batch
        print("Image processed for prediction")
        predictions = model.predict(img)
        print(f"Raw model predictions: {predictions}")
        predicted_class = np.argmax(predictions, axis=1)[0]
        categories = list(CATEGORY_SUGGESTIONS.keys())
        result = categories[predicted_class]
        print(f"Predicted class: {result}")
        return result
    except Exception as e:
        print(f"Error in model_predict: {e}")
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
