
# 🎵 Spotify Music Recommendation Using SparkML

This project implements a music recommendation system using Apache Spark's MLlib. Leveraging a dataset of Spotify tracks, we train and evaluate a collaborative filtering model to suggest music based on user preferences.

## 🔍 Overview

The primary goal of this project is to build a scalable recommendation engine using **Spark MLlib's ALS (Alternating Least Squares)** algorithm. It demonstrates data preprocessing, model training, evaluation, and generation of top-N recommendations for users.

## 📁 Dataset

The dataset contains features like:

- Track ID
- Track Name
- Popularity
- Artist Name
- Genre
- User ID
- User Ratings (for simulation)

You can find the dataset under the `data/` folder.

## ⚙️ Technologies Used

- **Apache Spark 3.x**
- **Spark MLlib (ALS algorithm)**
- **PySpark**
- **Python 3.x**
- **Jupyter Notebook**

## 🧪 Project Structure

```

Spotify-Music-Recommendation-Using-SparkML/
│
├── data/                        # Dataset files
├── notebooks/
│   └── Spotify\_Recommendation.ipynb   # Main implementation in Jupyter
├── results/                    # Output files and recommendation samples
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

````

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Swetha3009/Spotify-Music-Recommendation-Using-SparkML.git
cd Spotify-Music-Recommendation-Using-SparkML
````

### 2. Set up environment

It's recommended to use a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Ensure that Apache Spark is installed and configured.

### 3. Run the notebook

You can run the notebook in Jupyter:

```bash
jupyter notebook notebooks/Spotify_Recommendation.ipynb
```

## 📊 Key Features

* Data preprocessing and cleaning using Spark DataFrames.
* Building a collaborative filtering model using ALS.
* Hyperparameter tuning and evaluation (RMSE).
* Generating personalized music recommendations for a user.
* Scalable design for large datasets.

## 📈 Evaluation Metric

* **Root Mean Square Error (RMSE)** is used to evaluate model performance.

## ✅ Sample Output

Top-10 track recommendations for a sample user are stored in `results/user_recommendations.csv`.

## 🔮 Future Improvements

* Integrate with real Spotify API for dynamic data.
* Add content-based filtering using track features like tempo, energy, etc.
* Deploy the model as a web service or integrate into a frontend.

