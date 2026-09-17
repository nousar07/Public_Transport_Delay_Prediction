import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/assam_transport_delay_dataset_final_v2.csv"

NLP_DATA_PATH = "data/assam_final_nlp_features_v2.csv"

VECTORIZER_PATH = "models/final_assam_tfidf_vectorizer_v2.pkl"

NLP_INFO_PATH = "models/final_assam_nlp_info_v2.pkl"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("FINAL ASSAM TRANSPORT NLP BACKEND")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# CREATE TRANSPORT INCIDENT TEXT
# ============================================================

print("\nCreating transport incident text...")


def create_transport_text(row):

    weather = str(row["weather_condition"]).lower()

    transport = str(row["transport_type"]).lower()

    start = str(row["starting_point"]).lower()

    destination = str(row["destination_point"]).lower()

    traffic = float(row["traffic_congestion_index"])

    holiday = int(row["holiday"])

    # Traffic category
    if traffic >= 75:
        traffic_level = "heavy traffic congestion"
    elif traffic >= 50:
        traffic_level = "moderate traffic congestion"
    elif traffic >= 25:
        traffic_level = "light traffic congestion"
    else:
        traffic_level = "low traffic"

    # Holiday information
    if holiday == 1:
        holiday_text = "holiday travel"
    else:
        holiday_text = "normal working day"

    text = (
        f"{transport} journey from {start} to {destination}. "
        f"Weather condition is {weather}. "
        f"Traffic condition is {traffic_level}. "
        f"Travel occurs during {holiday_text}."
    )

    return text


df["transport_report_text"] = df.apply(
    create_transport_text,
    axis=1
)


# ============================================================
# DISPLAY SAMPLE TEXT
# ============================================================

print("\nSample generated NLP text:")

for i in range(min(5, len(df))):
    print(f"\n{i + 1}. {df.loc[i, 'transport_report_text']}")


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING TF-IDF FEATURES")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    df["transport_report_text"]
)

print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# ============================================================
# CREATE NLP FEATURE DATAFRAME
# ============================================================

feature_names = [
    f"nlp_{name}"
    for name in vectorizer.get_feature_names_out()
]

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)


# ============================================================
# COMBINE NLP FEATURES WITH BASIC IDENTIFIER
# ============================================================

nlp_output = pd.concat(
    [
        df[["route_id"]],
        tfidf_df
    ],
    axis=1
)


# ============================================================
# SAVE NLP FEATURES
# ============================================================

nlp_output.to_csv(
    NLP_DATA_PATH,
    index=False
)


# ============================================================
# SAVE TF-IDF VECTORIZER
# ============================================================

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)


# ============================================================
# SAVE NLP INFORMATION
# ============================================================

nlp_info = {
    "dataset": DATA_PATH,
    "output_dataset": NLP_DATA_PATH,
    "vectorizer": "TF-IDF",
    "max_features": 100,
    "ngram_range": (1, 2),
    "feature_count": len(feature_names),
    "rows": len(df),
    "text_column": "transport_report_text",
    "backend_only": True,
    "used_directly_for_prediction": False
}

with open(NLP_INFO_PATH, "wb") as f:
    pickle.dump(nlp_info, f)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("NLP BACKEND COMPLETED")
print("=" * 70)

print(f"\nOriginal dataset rows : {len(df)}")
print(f"NLP feature rows      : {len(nlp_output)}")
print(f"NLP feature count     : {len(feature_names)}")

print("\nSaved files:")

print(f"1. {NLP_DATA_PATH}")
print(f"2. {VECTORIZER_PATH}")
print(f"3. {NLP_INFO_PATH}")


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(df) == 5000:
    print("\n✓ 5000 rows confirmed")

if len(nlp_output) == 5000:
    print("✓ NLP rows confirmed")

if len(feature_names) > 0:
    print("✓ TF-IDF features created")

if os.path.exists(VECTORIZER_PATH):
    print("✓ TF-IDF vectorizer saved")

print("\n✓ NLP component is backend-only")

print("\nStep 9 completed successfully.")