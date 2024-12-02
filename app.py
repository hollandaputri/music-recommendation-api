from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from surprise import Dataset, Reader

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Load model yang telah dilatih
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
data = pd.read_csv("dataset.csv")

# Endpoint utama untuk cek status API
@app.get("/")
def read_root():
    return {"message": "Music Recommendation API is running!"}

# Fungsi untuk memberikan rekomendasi
def get_recommendations(user_id, model, data, n_recommendations=10):
    # Validasi apakah user_id ada dalam dataset
    if user_id not in data["user_id"].unique():
        raise HTTPException(status_code=404, detail="User ID tidak ditemukan!")

    all_items = data["id"].unique()
    user_items = data[data["user_id"] == user_id]["id"].unique()
    items_to_predict = [item for item in all_items if item not in user_items]

    # Prediksi rating untuk semua item
    predictions = [model.predict(user_id, item) for item in items_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Ambil n rekomendasi teratas
    top_recommendations = predictions[:n_recommendations]
    recommendations = []
    for pred in top_recommendations:
        song_details = data[data["id"] == pred.iid][["name", "artist"]].drop_duplicates().iloc[0]
        recommendations.append({
            "song_id": pred.iid,
            "song_name": song_details["name"],
            "artist": song_details["artist"],
            "predicted_rating": round(pred.est, 2)
        })

    return recommendations

# Endpoint untuk mendapatkan rekomendasi
@app.get("/recommend/")
def recommend(user_id: str, n_recommendations: int = 10):
    recommendations = get_recommendations(user_id, model, data, n_recommendations)
    return {"user_id": user_id, "recommendations": recommendations}
