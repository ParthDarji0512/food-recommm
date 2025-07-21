import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

preds = model.predict(dtrain)
mse = mean_squared_error(y, preds)
print(f"MSE: {mse}")


# Load data
data = pd.read_csv('recipe_final_with_nutrition.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Generate embeddings
X = np.vstack([get_bert_embedding(text) for text in data['ingredients_list']])
y = data['recipe_id'] if 'recipe_id' in data.columns else np.arange(len(data))  # dummy targets

# Train model
dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror"}
model = xgb.train(params, dtrain, num_boost_round=100)

# Save
model.save_model("model.xgb")
