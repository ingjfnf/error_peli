import pandas as pd
import sys
import joblib
import os

# Definir las palabras de parada
stop_words = set(["a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"])

def clean_text(text):
    # Función para la limpieza de texto
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_movie_genre(year, title, plot):
    # Cargar dataTesting
    dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip')
    
    # Buscar la fila que coincide con year, title y plot
    row = dataTesting[(dataTesting['year'] == year) & (dataTesting['title'] == title) & (dataTesting['plot'] == plot)]
    
    if row.empty:
        raise ValueError("No se encontró ninguna fila que coincida con los parámetros proporcionados.")
    
    # Carga los modelos y transformadores
    model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))
    vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'))
    mlb = joblib.load(os.path.join(os.path.dirname(__file__), 'mlb.pkl'))
    
    # Limpieza y vectorización del texto
    plot_clean = clean_text(plot)
    plot_vectorized = vectorizer.transform([plot_clean])
    
    # Predicción de géneros
    genre_predictions = model.predict(plot_vectorized)
    genres_formatted = mlb.inverse_transform(genre_predictions)
    
    return title, year, genres_formatted
