# 🚍 Assam Public Transport Delay Prediction

A machine learning and NLP-based project for predicting the expected arrival delay of public transport services operating on verified Assam routes.

## 📌 Project Overview

Public transport delays can be caused by several factors such as traffic congestion, weather conditions, route distance, holidays, and time of travel.

This project develops a **Public Transport Delay Prediction System** that uses machine learning to estimate the expected arrival delay of a bus journey based on journey and environmental conditions.

The project also includes an **NLP backend** that processes transport-related textual information using TF-IDF. The NLP component is designed as a backend extension and is not directly used as an input by the current prediction interface.

## 🎯 Objectives

* Predict expected public transport arrival delay in minutes.
* Use verified Assam bus routes and route distances.
* Consider factors such as:

  * Starting point
  * Destination
  * Route
  * Road distance
  * Date and time
  * Weather condition
  * Temperature
  * Traffic congestion
  * Holiday status
* Provide an easy-to-use prediction interface using Streamlit.
* Integrate NLP techniques for transport-related textual data processing.
* Build a practical data science project focused on a real-world transportation problem.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **TF-IDF**
* **Gradient Boosting**
* **Random Forest**
* **Extra Trees**
* **Linear Regression**
* **Streamlit**
* **Open-Meteo API**

## 🤖 Machine Learning

Multiple regression models were trained and compared:

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Linear Regression | 2.1855 | 2.6983 | 0.5289 |
| Random Forest     | 1.9124 | 2.3752 | 0.6350 |
| Gradient Boosting | 1.7553 | 2.2120 | 0.6834 |
| Extra Trees       | 2.0780 | 2.5955 | 0.5641 |

The **Gradient Boosting Regressor** was selected based on the lowest MAE among the tested models.

> Note: These results are based on the project's generated/simulated training dataset and should not be interpreted as official ASTC prediction accuracy.

## 🧠 NLP Component

The project includes an NLP backend for processing transport-related textual reports.

The NLP pipeline uses:

1. Text preprocessing
2. TF-IDF vectorization
3. Unigram and bigram features
4. A maximum vocabulary size of 100 features

The generated NLP features are stored separately and can be used for future improvements to the prediction system.

## 📊 Dataset

The project contains:

* Verified Assam bus route master
* Public transport delay dataset
* NLP feature dataset

The route master contains verified service-level routes with route information and road distances.

The current dataset contains **5,000 simulated journey records across 16 verified bus route records**.

### Important Data Note

The delay values in the current training dataset are **simulated/generated values**, not official historical ASTC delay records.

Therefore, the model demonstrates the complete machine-learning workflow and application architecture, but its predictions should not be treated as real-time official ASTC predictions.

## 🗂️ Project Structure

```text
Public_Transport_Delay_Prediction/
│
├── app.py
├── generate_dataset.py
├── prepare_dataset.py
├── train_models.py
├── NLP_backend.py
├── .gitignore
│
├── data/
│   ├── assam_routes_final_verified.csv
│   ├── assam_transport_delay_dataset_final_v2.csv
│   └── assam_final_nlp_features_v2.csv
│
└── models/
    ├── final_assam_delay_model_v2.pkl
    ├── final_assam_model_columns_v2.pkl
    ├── final_assam_model_info_v2.pkl
    └── final_assam_tfidf_vectorizer_v2.pkl
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nousar07/Public_Transport_Delay_Prediction.git
```

### 2. Open the project folder

```bash
cd Public_Transport_Delay_Prediction
```

### 3. Install the required packages

```bash
pip install pandas numpy scikit-learn nltk streamlit requests
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 🌡️ Temperature Data

The application can retrieve temperature information dynamically using the **Open-Meteo weather service** based on the selected starting location.

If the weather service is unavailable, the application uses its fallback temperature handling.

## 🚌 Current Route Coverage

The current verified route master focuses on **ASTC bus service routes** connecting locations such as:

* Guwahati
* Nagaon
* Jorhat
* Dibrugarh
* Haflong
* Margherita
* Tinsukia
* Biswanath Chariali

More verified routes can be added as reliable service-level route information becomes available.

## 🔮 Future Scope

* Add more verified Assam public transport routes.
* Incorporate real historical transport delay data.
* Integrate live traffic information.
* Improve real-time weather integration.
* Use NLP-generated incident information directly in model prediction.
* Add more transport types after obtaining verified route and service data.
* Develop a mobile-friendly version of the application.
* Deploy the application as a cloud-based prediction service.

## 👨‍💻 Project Information

**Project:** Assam Public Transport Delay Prediction
**Domain:** Data Science & Natural Language Processing
**Platform:** Python & Streamlit
**Project Type:** Final-Year Industrial Internship Project
