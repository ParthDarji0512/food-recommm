
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

data = pd.read_csv("recipe_final_with_nutrition.csv")

def clean_ingredients(text):
    return re.sub(r'\s+', ' ', str(text).lower().strip())

data['ingredients_list'] = data['ingredients_list'].fillna('').apply(clean_ingredients)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])


knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X_ingredients)

def recommend_recipes(ingredients_str):
    """Get top 5 recommended recipes based on ingredients"""
    input_transformed = vectorizer.transform([ingredients_str])
    distances, indices = knn.kneighbors(input_transformed)
    top_recipes = data.iloc[indices[0]]
    return top_recipes[['recipe_name', 'ingredients_list', 'image_url',
                        'calories', 'carbohydrates', 'protein', 'cholesterol']]

def truncate(text, length=50):
    """Truncate text to desired length"""
    return text[:length] + "..." if len(text) > length else text

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
        results = model(img)
        detected_items = results.pandas().xywh[0]
        ingredients = list(set(detected_items['name'].tolist()))
        ingredients_str = ', '.join(ingredients)

        recipes = recommend_recipes(ingredients_str)
        response = {
            "ingredients": ingredients,
            "recipes": recipes.to_dict(orient='records')
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        recipes = recommend_recipes(ingredients)
        recommendations = recipes.to_dict(orient='records')

    return render_template('index.html',
                           recommendations=recommendations,
                           truncate=truncate)

if __name__ == '__main__':
    app.run(debug=True)
