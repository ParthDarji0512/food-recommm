# Recipe-Recommendation-Food-Recommendation-Python-Machine-Learning

# 🍽️ AI-Powered Recipe Recommendation App

This project is an AI-driven recipe recommendation system that suggests the top 5 most relevant recipes based on ingredients detected from an uploaded image or user input. It combines computer vision (YOLOv5), natural language processing (TF-IDF), and similarity matching (KNN) to provide healthy and tailored recipe recommendations.

---

## 🔍 Features

- 🍲 Detects ingredients from uploaded food images using YOLOv5
- 📑 Accepts manual ingredient input as an alternative
- 🔎 Recommends top 5 similar recipes using TF-IDF + KNN
- 🥗 Shows detailed recipe info: calories, protein, carbs, and more
- 🌐 Built using Flask and deployable as a web app

---

## 🧠 Tech Stack

- **Frontend**: HTML (Jinja2 templating)
- **Backend**: Python (Flask)
- **ML Models**:
  - [YOLOv5](https://github.com/ultralytics/yolov5) (for ingredient detection)
  - TF-IDF Vectorizer + KNN (for recipe recommendation)
- **Libraries**: `torch`, `Pillow`, `pandas`, `scikit-learn`

---

## 📁 Project Structure

├── app.py # Main Flask app
├── recipe_final_with_nutrition.csv # Dataset with recipes & nutrition
├── templates/
│ └── index.html # Frontend page

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/recipe-recommender.git
   cd recipe-recommender
2.Install dependencies:
  pip install -r requirements.txt
3.Ensure the dataset recipe_final_with_nutrition.csv is present in the root directory.
4.Run the Flask app: python app.py
5.Access the app: Open http://localhost:5000 in your browser.


📸 Image-based Recommendation
Upload an image containing ingredients (e.g., tomatoes, carrots, etc.)

The app uses YOLOv5 to detect visible items.

Detected ingredients are used to recommend recipes.

✍️ Manual Input
Alternatively, enter ingredients as text on the homepage to get recipe suggestions.

📊 Output Format
Each recommendation includes:

✅ Recipe name

🍴 Ingredients (shortened)

🔗 Image URL (if available)

🔢 Nutrition: Calories, Carbs, Protein, Cholesterol

📌 Notes
This app uses a pretrained YOLOv5s model via torch.hub.

Accuracy may vary based on image clarity and angle.

You can expand the dataset and improve vectorization for better results.

A virtual environment is must required for running this project if any problem persist while running we can connect.
