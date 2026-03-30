# MSE 446 Project — Team 2: GMM-Based Music Recommendation

A machine learning system that recommends songs from smaller/underground artists
based on a user's favourite popular songs, using Gaussian Mixture Models (GMM)
and cosine similarity.

## Team Members
Sam Gupta, Aysha Hide, Jacqueline Pang, Hiro Chen, Louise Lee

## Repository Structure
```
├── MSE446_Group2_DataCleaning_C.ipynb   # Data cleaning & preprocessing
├── MSE446_Group2_GMMModel_C.ipynb       # GMM model, evaluation & recommendations
├── datasets/
│   ├── songs_cleaned.csv                # Cleaned dataset (output of data cleaning notebook)
│   ├── high_popularity_spotify_data.csv # Raw popular songs
│   ├── low_popularity_spotify_data.csv  # Raw small-artist songs
│   └── spotify_data_artists.csv         # Artist metadata
├── requirements.txt                     # Python dependencies
└── README.md
```

## Setup & Installation

1. **Python version:** 3.9 or higher recommended.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks in order:**
   - First: `MSE446_Group2_DataCleaning_C.ipynb` — reads raw CSVs from `datasets/`, produces `datasets/songs_cleaned.csv`
   - Then: `MSE446_Group2_GMMModel_C.ipynb` — loads `datasets/songs_cleaned.csv`, trains the GMM, runs evaluation, and demonstrates recommendations

## How to Reproduce Results

1. Open `MSE446_Group2_GMMModel_C.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells sequentially (Kernel → Restart & Run All).
3. The notebook will:
   - Split data into train/validation/test sets
   - Tune GMM hyperparameters via BIC grid search on the training set
   - Evaluate clustering quality (Silhouette Score, Davies-Bouldin Index)
   - Demonstrate song recommendations from the final model

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn
