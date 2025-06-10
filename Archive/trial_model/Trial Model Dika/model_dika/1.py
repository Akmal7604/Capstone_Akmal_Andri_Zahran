from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import re
import numpy as np
import xgboost as xgb  # Impor xgboost

app = FastAPI()

# --- 1. Load Model and Preprocessing Objects ---
try:
    # Load model menggunakan XGBoost
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model('xgb_model.json')  # Load model dari format JSON
    
    with open('mlb.pkl', 'rb') as f:
        loaded_mlb = pickle.load(f)
    with open('le.pkl', 'rb') as f:
        loaded_le = pickle.load(f)
    with open('model_input_feature_columns.pkl', 'rb') as f:
        loaded_model_input_feature_columns = pickle.load(f)
    with open('bins_labels.pkl', 'rb') as f:
        bin_data = pickle.load(f)
        bins = bin_data['bins']
        labels = bin_data['labels']

    print("Model and preprocessing objects loaded successfully.")

    # Load dataset
    try:
        data = pd.read_csv('../recipes_new.csv')
        #data = pd.read_csv('recipes_new.csv')
        columns = [
            'RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime',
            'RecipeIngredientParts', 'Calories', 'FatContent', 'SaturatedFatContent',
            'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
            'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeServings', 'Keywords',
            'Images'
        ]
        data = data[columns]
        data.rename(columns={'Keywords': 'tags'}, inplace=True)
        data['tags'] = data['tags'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z, ]', '', x))
        data['tags'] = data['tags'].apply(lambda x: [tag.strip() for tag in x.lower().split(',') if tag.strip()])
        
        print("CSV data loaded and processed successfully.")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise RuntimeError(f"CSV load error: {e}")

except Exception as e:
    print(f"Error loading files: {e}")
    raise RuntimeError(f"Initialization failed: {e}")

# --- 2. Recommendation Function ---
def search_food_by_keywords(df, keywords, target_calories, top_n=5):
    keywords = [k.lower() for k in keywords]
    temp_df = df.copy()
    temp_df['tags_cleaned'] = temp_df['tags'].copy()
    
    # Filter berdasarkan keyword
    filtered_df = temp_df[
        temp_df['tags_cleaned'].apply(lambda x: any(k in x for k in keywords))
    ].copy()
    
    if filtered_df.empty:
        return pd.DataFrame(columns=['RecipeId', 'food', 'tags', 'calories', 'pred_prob'])

    # Tentukan calorie bin
    target_calorie_label = None
    if target_calories <= bins[0]:
        target_calorie_label = labels[0]
    elif target_calories >= bins[-1]:
        target_calorie_label = labels[-1]
    else:
        for i in range(len(bins) - 1):
            if bins[i] < target_calories <= bins[i+1]:
                target_calorie_label = labels[i]
                break

    try:
        target_calorie_bin_encoded = loaded_le.transform([target_calorie_label])[0]
    except ValueError:
        print(f"Target calorie bin error: {target_calorie_label}")
        return pd.DataFrame(columns=['RecipeId', 'food', 'tags', 'calories', 'pred_prob'])

    # Siapkan fitur untuk prediksi
    filtered_keywords_encoded = loaded_mlb.transform(filtered_df['tags_cleaned'])
    filtered_keywords_df = pd.DataFrame(
        filtered_keywords_encoded, 
        columns=loaded_mlb.classes_, 
        index=filtered_df.index
    )

    X_predict = pd.DataFrame(
        0, 
        index=filtered_df.index, 
        columns=loaded_model_input_feature_columns
    )

    # Isi fitur numerik
    numerical_features = [
        'CookTime', 'PrepTime', 'TotalTime', 'FatContent', 'SaturatedFatContent',
        'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent',
        'SugarContent', 'ProteinContent', 'RecipeServings'
    ]
    
    for col in numerical_features:
        if col in filtered_df.columns:
            X_predict[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

    # Isi fitur tag
    for col in loaded_mlb.classes_:
        if col in filtered_keywords_df.columns:
            X_predict[col] = filtered_keywords_df[col]

    # Prediksi probabilitas
    try:
        pred_proba = loaded_model.predict_proba(X_predict)
        filtered_df['pred_prob'] = pred_proba[:, target_calorie_bin_encoded]
    except Exception as e:
        print(f"Prediction error: {e}")
        return pd.DataFrame(columns=['RecipeId', 'food', 'tags', 'calories', 'pred_prob'])

    # Urutkan dan kembalikan hasil terbaik
    filtered_df = filtered_df.sort_values(by='pred_prob', ascending=False)
    
    return filtered_df[[
        'RecipeId', 'Name', 'tags', 'Calories', 'pred_prob'
    ]].rename(columns={'Name': 'food'}).head(top_n)

# --- 3. Pydantic Models ---
class Recommendation(BaseModel):    
    RecipeId: int
    food: str
    tags: List[str]
    Calories: float
    pred_prob: float

class RecommendationsResponse(BaseModel):
    recommendations: List[Recommendation]

# 
class getById():
    

# --- 4. API Endpoint ---
@app.get("/recommend_recipes", response_model=RecommendationsResponse)
async def recommend_recipes(
    keywords: str = Query(..., description="Comma-separated keywords (e.g., 'chicken,easy,dinner')"),
    target_calories: float = Query(..., gt=0, description="Target calorie amount"),
    top_n: Optional[int] = Query(5, ge=1, le=20, description="Number of results")
):
    try:
        keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
        if not keyword_list:
            raise HTTPException(400, "Keywords cannot be empty")

        results = search_food_by_keywords(data, keyword_list, target_calories, top_n)
        
        if results.empty:
            return {"recommendations": []}
        
        # Konversi tags dari list menjadi list string
        results['tags'] = results['tags'].apply(lambda x: [str(tag) for tag in x])
        
        return {
            "recommendations": results.to_dict(orient='records')
        }
        
    except Exception as e:
        raise HTTPException(500, f"Internal error: {str(e)}")

# 5. API Endpoint by id
@app.get("/id", )