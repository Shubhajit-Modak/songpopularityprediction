import joblib
import pandas as pd


# ============================================================
# LOAD SAVED MODEL
# ============================================================

model_package = joblib.load("models/best_model.pkl")

model = model_package["model"]
features = model_package["features"]


print("=" * 55)
print("SPOTIFY SONG POPULARITY PREDICTOR")
print("=" * 55)

print("\nRandom Forest model loaded successfully!")


# ============================================================
# INPUT VALIDATION
# ============================================================

def get_float(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(
                f"Please enter a value between "
                f"{minimum} and {maximum}."
            )

        except ValueError:
            print("Please enter a valid number.")


# ============================================================
# ENTER SONG FEATURES
# ============================================================

print("\nEnter the audio features of the song:\n")

danceability = get_float(
    "Danceability (0-1): ",
    0.0,
    1.0
)

energy = get_float(
    "Energy (0-1): ",
    0.0,
    1.0
)

loudness = get_float(
    "Loudness (dB): ",
    -60.0,
    2.0
)

speechiness = get_float(
    "Speechiness (0-1): ",
    0.0,
    1.0
)

acousticness = get_float(
    "Acousticness (0-1): ",
    0.0,
    1.0
)

instrumentalness = get_float(
    "Instrumentalness (0-1): ",
    0.0,
    1.0
)

liveness = get_float(
    "Liveness (0-1): ",
    0.0,
    1.0
)

valence = get_float(
    "Valence (0-1): ",
    0.0,
    1.0
)

tempo = get_float(
    "Tempo (BPM): ",
    40.0,
    250.0
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

new_song = pd.DataFrame(
    [[
        danceability,
        energy,
        loudness,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo
    ]],
    columns=features
)


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(new_song)[0]

probabilities = model.predict_proba(new_song)[0]

class_probabilities = dict(
    zip(model.classes_, probabilities)
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 55)
print("PREDICTION RESULT")
print("=" * 55)

print(f"Predicted Category: {prediction}")

print("\nPrediction probabilities:")

for category, probability in class_probabilities.items():
    print(f"{category}: {probability:.2%}")


print("\nPrediction completed successfully!")