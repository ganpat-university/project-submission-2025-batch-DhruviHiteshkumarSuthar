import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import logging
import mysql.connector
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import random
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(level=logging.DEBUG)


# To generate a random API token, you can use the following code snippet:
# import secrets
# token = secrets.token_hex(32)
# print(token)
# Save the token in a .env file as API_TOKEN=<your_token>

from dotenv import load_dotenv
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")


model = tf.keras.models.load_model("emotion_recognition_cnn_rnn_model_final.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Error loading Haar cascade classifier.")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Thoughtful movie recommendations
with open("movies.json", "r") as f:
    emotion_to_movies = json.load(f)

omdb_api_key = "3c2ac814"

application = Flask(__name__)
CORS(application)

# MySQL RDS config
db_config = {
    "host": "emotion-insights-logs.cz6wwus4stt4.ap-south-1.rds.amazonaws.com",
    "user": "admin",
    "password": "Dhruvi#123",
    "database": "fer"
}

# Log to MySQL
def log_prediction_to_db(emotion, confidence):
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor()
            query = "INSERT INTO emotion_logs (emotion, confidence) VALUES (%s, %s)"
            cursor.execute(query, (emotion, confidence))
            connection.commit()
            cursor.close()
            connection.close()
    except mysql.connector.Error as e:
        logging.error(f"MySQL error: {e}")

# Movie data from OMDB
def get_movie_recommendation(emotion):
    movie_list = emotion_to_movies.get(emotion, emotion_to_movies["Neutral"])
    selected_movie = random.choice(movie_list)
    url = f"http://www.omdbapi.com/?t={selected_movie}&apikey={omdb_api_key}"

    try:
        response = requests.get(url)
        data = response.json()
        if data.get("Response") == "True":
            return {
                "title": data.get("Title"),
                "year": data.get("Year"),
                "poster": data.get("Poster"),
                "description": data.get("Plot"),
                "trailer_link": f"https://www.youtube.com/results?search_query={data.get('Title').replace(' ', '+')}+trailer"
            }
    except Exception as e:
        logging.error(f"OMDB error: {e}")
    return None

application.wsgi_app = ProxyFix(application.wsgi_app, x_for=1)
limiter = Limiter(key_func=get_remote_address, app=application)

# Apply rate limiting to the /predict endpoint
# @application.route("/predict", methods=["POST"])
# @limiter.limit("2 per 30 seconds")  # 2 request per 30 seconds per IP
# def predict():
#     try:
#         data = request.json
#         image_data = base64.b64decode(data["image"])
#         np_arr = np.frombuffer(image_data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
#         resized = cv2.resize(frame, (48, 48)) / 255.0
#         face = np.expand_dims(resized, axis=(0, -1))
#         prediction = model.predict(face)
#         predicted_index = np.argmax(prediction)
#         emotion = emotion_labels[predicted_index]
#         confidence = float(np.max(prediction))
#         log_prediction_to_db(emotion, confidence)
#         return jsonify({"emotion": emotion, "confidence": confidence})
#     except Exception as e:
#         logging.exception("Predict error")
#         return jsonify({"error": str(e)}), 500

@application.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")  # 4 request per 60 seconds per IP
def predict():
    try:
        data = request.json
        image_data = base64.b64decode(data["image"])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # Use the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        # Resize to 48x48
        face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA if face.shape[0] >= 48 else cv2.INTER_LINEAR)
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=(0, -1))

        prediction = model.predict(face_expanded)
        predicted_index = np.argmax(prediction)
        emotion = emotion_labels[predicted_index]
        confidence = float(np.max(prediction))

        log_prediction_to_db(emotion, confidence)
        return jsonify({"emotion": emotion, "confidence": confidence})

    except Exception as e:
        logging.exception("Predict error")
        return jsonify({"error": str(e)}), 500

# API: Get logs
@application.route("/logs", methods=["GET"])
def get_logs():
    token = request.headers.get("X-API-TOKEN")
    if token != API_TOKEN:
        return jsonify({"error": "Unauthorized access"}), 401

    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, timestamp, emotion, confidence FROM emotion_logs ORDER BY timestamp DESC LIMIT 20")
        logs = cursor.fetchall()
        cursor.close()
        connection.close()
        return jsonify({"logs": logs})
    except:
        return jsonify({"error": "Log fetch failed"}), 500


# API: Get movie for emotion
@application.route("/movie_recommendation", methods=["GET"])
def movie_recommendation():
    emotion = request.args.get("emotion")
    if not emotion:
        return jsonify({"error": "Emotion not provided"}), 400
    movie = get_movie_recommendation(emotion)
    if movie:
        return jsonify(movie)
    return jsonify({"error": "No movie found"}), 404

# ---------------- Dash UI ----------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, server=application, url_base_pathname='/dashboard/', external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    # Navbar
    dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Img(src="./assets/newlogo.png", height="48px", style={"marginRight": "10px"}),
                    html.Span("Emotion Insights", style={
                        "fontSize": "24px", "fontWeight": "600", "color": "white",
                        "verticalAlign": "middle"
                    })
                ], style={"display": "flex", "alignItems": "center"})
            )
        ], align="center", className="g-0")
    ]),
    style={"background": "linear-gradient(135deg, #b19cd9, #89CFF0)", "padding": "8px 16px"},
    dark=False,
    className="mb-4 shadow-sm"
    ),

    # Main content row
    dbc.Row([
        # Left: Video Feed
        dbc.Col([
            html.Div([
                html.H5("Live Video Feed", className="text-center mb-3"),
                html.Video(id="video-feed", autoPlay=True, muted=True, style={
                    "width": "100%", "borderRadius": "12px", "objectFit": "cover",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"
                }),
            ], className="p-4 bg-white rounded shadow-sm h-100")
        ], xs=12, md=6),

        # Right: Emotion and Movie Recommendation
        dbc.Col([
            # Current Emotion Box
            html.Div([
                html.H5("Current Emotion", className="text-center mb-3"),
                html.H2(id="current-emotion", className="text-center text-primary", style={
                    "fontSize": "clamp(20px, 2vw, 28px)", "fontWeight": "bold"
                }),
            ], className="p-4 bg-white rounded shadow-sm mb-4"),

            # Movie Recommendation Box
            html.Div([
                html.H5(id="recommendation-header", className="text-center mb-3", style={
                    "color": "#6a0dad", "fontWeight": "600"
                }),  # This should say: Why don't you try watching <movie>?

                dbc.Row([
                    dbc.Col(html.Img(id="movie-poster", style={
                        "width": "100%", "borderRadius": "10px", "maxWidth": "150px"
                    }), width=4),

                    dbc.Col([
                        html.P(id="movie-description", style={"fontSize": "14px"}),
                        html.A("▶ Watch Trailer", id="movie-trailer", href="#", target="_blank", style={
                            "color": "#007bff", "textDecoration": "none", "fontWeight": "500", "display": "none"
                        })
                    ], width=8)
                ])
            ], className="p-4 bg-white rounded shadow-sm")
        ], xs=12, md=6)
    ], className="mb-4"),

    # Intervals and dummy output
    dcc.Interval(id="startup-interval", interval=3000, n_intervals=0, max_intervals=1),
    dcc.Interval(id="update-interval", interval=15000, n_intervals=0),
    html.Div(id="dummy-output", style={"display": "none"})
], fluid=True, style={"backgroundColor": "#ffffff", "minHeight": "100vh", "paddingTop": "24px"})



# JS callback (client-side)
app.clientside_callback(
    """
    function(n1, n2) {
        var video = document.getElementById('video-feed');
        if (!video || video.videoWidth === 0) return "";
        var canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        var base64Image = canvas.toDataURL("image/jpeg").split(",")[1];

        fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ image: base64Image })
        })
        .then(r => r.json())
        .then(data => {
            if (data.emotion) {
                document.getElementById("current-emotion").innerText = data.emotion;
                fetch(`/movie_recommendation?emotion=${data.emotion}`)
                    .then(r => r.json())
                    .then(movie => {
                        if (movie.title) {
                            document.getElementById("recommendation-header").innerText = `Why don’t you try watching \"${movie.title}\"?`;
                            document.getElementById("movie-description").innerText = movie.description;
                            document.getElementById("movie-poster").src = movie.poster;
                            document.getElementById("movie-trailer").href = movie.trailer_link;
                            document.getElementById("movie-trailer").style.display = "inline";
                        }
                    });
            }
        })
        .catch(console.error);
        return "";
    }
    """,
    Output("dummy-output", "children"),
    [Input("startup-interval", "n_intervals"), Input("update-interval", "n_intervals")]
)

@application.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port, debug=True)