from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import re
import numpy as np

# --- 1. Load the necessary pre-trained objects and data ---
try:
    loaded_model = pickle.load(open('xgb_model.pkl', 'rb'))
    with open('mlb.pkl', 'rb') as f:
        loaded_mlb = pickle.load(f)
    with open('le.pkl', 'rb') as f:
        loaded_le = pickle.load(f)
    with open('model_input_feature_columns.pkl', 'rb') as f:
        loaded_model_input_feature_columns = pickle.load(f)

    data = pd.read_csv('../recipes_new.csv')
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

    bins = [0, 200, 400, 600, 800, 1000, np.inf]
    labels = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1000+']

    print("All necessary objects and data loaded successfully for FastAPI app.")

except Exception as e:
    print(f"Error loading required files: {e}. Please ensure 'xgb_model.pkl', 'mlb.pkl', 'le.pkl', 'model_input_feature_columns.pkl' and 'recipes_new.csv' are in the correct directories.")
    raise RuntimeError(f"Failed to load essential files: {e}")

# --- 2. Define the search_food_by_keywords function ---
def search_food_by_keywords(df, keywords, target_calories, top_n=5):
    keywords = [k.lower() for k in keywords]
    temp_df = df.copy()
    temp_df['tags_cleaned'] = temp_df['tags'].copy()
    filtered_df = temp_df[temp_df['tags_cleaned'].apply(lambda x: any(k in x for k in keywords))].copy()

    if filtered_df.empty:        
        return pd.DataFrame(columns=['RecipeId', 'food', 'tags', 'calories', 'pred_prob'])

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
        print(f"Error: Target calorie label '{target_calorie_label}' not found in loaded LabelEncoder classes.")
        return pd.DataFrame(columns=['RecipeId', 'food', 'tags', 'calories', 'pred_prob'])

    filtered_keywords_encoded = loaded_mlb.transform(filtered_df['tags_cleaned'])
    filtered_keywords_df = pd.DataFrame(filtered_keywords_encoded, columns=loaded_mlb.classes_, index=filtered_df.index)

    X_predict = pd.DataFrame(0, index=filtered_df.index, columns=loaded_model_input_feature_columns)

    numerical_features_for_predict = [
        'CookTime', 'PrepTime', 'TotalTime',
        'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
        'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
        'RecipeServings'
    ]

    for col in numerical_features_for_predict:
        if col in filtered_df.columns:
            X_predict[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

    for col in loaded_mlb.classes_:
        if col in filtered_keywords_df.columns:
            X_predict[col] = filtered_keywords_df[col]

    pred_proba = loaded_model.predict_proba(X_predict)

    if target_calorie_bin_encoded >= pred_proba.shape[1]:
        print(f"Error: Encoded target calorie bin {target_calorie_bin_encoded} is out of bounds for model's prediction probabilities (num_classes={pred_proba.shape[1]}).")
        return pd.DataFrame(columns=['RecipeId', 'food', 'tags', 'calories', 'pred_prob'])

    filtered_df['pred_prob'] = [prob[target_calorie_bin_encoded] for prob in pred_proba]

    filtered_df = filtered_df.sort_values(by='pred_prob', ascending=False)

    # *** MODIFICATION HERE: Include 'RecipeId' ***
    return filtered_df[['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob']].rename(
        columns={'Name': 'food'}
    ).head(top_n)

# --- 3. Initialize FastAPI App ---
app = FastAPI()

# --- Define Pydantic Models for Response ---
class Recommendation(BaseModel):    
    RecipeId: int # Or int, depending on your data type
    food: str
    tags: List[str]
    Calories: float
    pred_prob: float

class RecommendationsResponse(BaseModel):
    recommendations: List[Recommendation]

# --- 4. Define API Endpoint with GET and Query Parameters ---
@app.get("/recommend_recipes", response_model=RecommendationsResponse)
async def recommend_recipes(
    keywords: str = Query(..., description="Daftar kata kunci yang dipisahkan koma (contoh: 'high protein,asian,low calorie')"),
    target_calories: float = Query(..., gt=0, description="Target jumlah kalori yang diinginkan (harus lebih dari 0)"),
    top_n: Optional[int] = Query(5, ge=1, description="Jumlah rekomendasi teratas yang akan dikembalikan (default: 5, min: 1)")
):
    """
    Merekomendasikan resep berdasarkan kata kunci dan target kalori.
    """
    try:
        parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        if not parsed_keywords:
            raise HTTPException(status_code=400, detail="Parameter 'keywords' tidak boleh kosong.")

        recommendations_df = search_food_by_keywords(
            data, parsed_keywords, target_calories, top_n
        )

        if recommendations_df.empty:
            return {"recommendations": []}
        else:
            return {"recommendations": recommendations_df.to_dict(orient='records')}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")

# --- 5. Jalankan Aplikasi FastAPI ---
# Untuk menjalankan aplikasi, simpan file ini sebagai `app.py` atau `main.py`.
# Kemudian di terminal, jalankan:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000