from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import re
import numpy as np
import json
import ast
from tensorflow.keras.models import load_model

app = FastAPI()

# --- 1. Load Model and Preprocessing Objects ---
try:
    # Load the trained TensorFlow model
    loaded_model = load_model('tensorflow_model.h5')

    # Load the preprocessing objects
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

    # Load and preprocess the dataset
    # Ganti path ini jika perlu
    data = pd.read_csv('../recipes_new.csv')
    
    # Preprocess the data similarly to the training script
    data.rename(columns={'Keywords': 'tags'}, inplace=True)
    data['tags'] = data['tags'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z, ]', '', x))
    data['tags'] = data['tags'].apply(lambda x: [tag.strip() for tag in x.lower().split(',') if tag.strip()])

    # Helper functions for image URL parsing
    def is_valid_image_url(url: Optional[str]) -> bool:
        if not url or not isinstance(url, str): return False
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG', '.JPEG', '.PNG']
        return any(url.lower().endswith(ext) for ext in image_extensions)

    def complete_image_url(url: str) -> str:
        base_url = "https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/"
        if url.startswith("http"): return url
        return base_url + url.lstrip('/')

    def parse_image_url(image_str):
        if not image_str or not isinstance(image_str, str): return None
        try:
            images = json.loads(image_str.replace("'", '"'))
            if isinstance(images, list) and images:
                for url in images:
                    if is_valid_image_url(url): return url
        except json.JSONDecodeError: pass
        try:
            images = ast.literal_eval(image_str)
            if isinstance(images, list) and images:
                for url in images:
                    if is_valid_image_url(url): return url
        except (ValueError, SyntaxError): pass
        for url in image_str.split(','):
            url = url.strip()
            if is_valid_image_url(url): return url
        return None

    data['PrimaryImage'] = data['Images'].apply(parse_image_url)

except Exception as e:
    print(f"Error during initialization: {e}")
    raise RuntimeError(f"Initialization failed: {e}")

# --- 2. Recommendation Function with RANKING & RANDOMIZATION ---
def search_food_by_keywords(df, keywords, target_calories, top_n=5):
    keywords = [k.lower() for k in keywords]
    temp_df = df.copy()
    temp_df['tags_cleaned'] = temp_df['tags'].copy()

    # Langkah 1: Dapatkan kandidat awal
    candidate_df = temp_df[
        temp_df['tags_cleaned'].apply(lambda tag_list: any(k in tag_list for k in keywords))
    ].copy()

    if candidate_df.empty: return pd.DataFrame()

    # Langkah 2: Hitung 'match_score'
    def calculate_match_score(tag_list):
        return sum(1 for k in keywords if k in tag_list)
    candidate_df['match_score'] = candidate_df['tags_cleaned'].apply(calculate_match_score)

    target_calorie_label = None
    if target_calories <= bins[0]: target_calorie_label = labels[0]
    elif target_calories >= bins[-1]: target_calorie_label = labels[-1]
    else:
        for i in range(len(bins) - 1):
            if bins[i] < target_calories <= bins[i + 1]:
                target_calorie_label = labels[i]
                break
    
    if target_calorie_label is None: return pd.DataFrame()
    try:
        target_calorie_bin_encoded = loaded_le.transform([target_calorie_label])[0]
    except ValueError: return pd.DataFrame()

    filtered_keywords_encoded = loaded_mlb.transform(candidate_df['tags_cleaned'])
    filtered_keywords_df = pd.DataFrame(filtered_keywords_encoded, columns=loaded_mlb.classes_, index=candidate_df.index)

    X_predict = pd.DataFrame(0, index=candidate_df.index, columns=loaded_model_input_feature_columns)
    numerical_features = ['CookTime', 'PrepTime', 'TotalTime', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeServings']
    for col in numerical_features:
        if col in candidate_df.columns: X_predict[col] = pd.to_numeric(candidate_df[col], errors='coerce').fillna(0)
    for col in loaded_mlb.classes_:
        if col in filtered_keywords_df.columns: X_predict[col] = filtered_keywords_df[col]

    try:
        X_predict_tf = X_predict.values.astype(np.float32)
        pred_proba = loaded_model.predict(X_predict_tf, verbose=0) # verbose=0 untuk mematikan log prediksi
        candidate_df['pred_prob'] = [prob[target_calorie_bin_encoded] for prob in pred_proba]
    except Exception as e:
        print(f"Prediction error: {e}")
        candidate_df['pred_prob'] = 0.0

    # === BLOK PERUBAHAN UTAMA UNTUK RANDOMISASI ===

    # Langkah 3: Urutkan untuk mendapatkan kandidat terbaik
    sorted_results = candidate_df.sort_values(by=['match_score', 'pred_prob'], ascending=[False, False])

    # Langkah 4: Buat 'kolam' kandidat terbaik yang lebih besar
    # Ambil minimal 20 resep atau 5x lipat dari top_n, mana yang lebih besar.
    # Tapi jangan lebih besar dari jumlah total hasil yang ditemukan.
    pool_size = min(len(sorted_results), max(top_n * 5, 20))
    candidate_pool = sorted_results.head(pool_size)

    # Langkah 5: Ambil secara acak dari 'kolam' tersebut.
    # Pastikan jumlah sample tidak lebih besar dari ukuran pool itu sendiri.
    num_to_sample = min(top_n, len(candidate_pool))
    if num_to_sample > 0:
        final_results = candidate_pool.sample(n=num_to_sample, random_state=None) # random_state=None agar selalu acak
    else:
        final_results = pd.DataFrame(columns=candidate_pool.columns) # Kembalikan dataframe kosong jika tidak ada yg bisa di-sample
    
    # === AKHIR BLOK PERUBAHAN ===

    output_columns = ['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent', 'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage', 'match_score']
    for col in output_columns:
        if col not in final_results.columns: final_results[col] = np.nan

    return final_results[output_columns].rename(columns={'Name': 'food'})

# --- 3. Pydantic Models (No Change) ---
class RecipeRecommendation(BaseModel):
    RecipeId: int; Name: str; Calories: float; ProteinContent: float; CarbohydrateContent: float; FatContent: float; ServingSize: int = 1; ServingUnit: str = "Porsi"; Image: Optional[str]

class RecommendationsResponse(BaseModel):
    recommendations: List[RecipeRecommendation]

class RecipeDetail(BaseModel):
    RecipeId: int; Name: str; CookTime: int; PrepTime: int; TotalTime: int; Image: Optional[str]; Keywords: List[str]; RecipeIngredientParts: List[str]; Calories: float; FatContent: float; SaturatedFatContent: float; CholesterolContent: float; SodiumContent: float; CarbohydrateContent: float; FiberContent: float; SugarContent: float; ProteinContent: float; ServingSize: int = 1; ServingUnit: str = "Porsi"

# --- 4. API Endpoints (No Change) ---
@app.get("/recommend_recipes", response_model=RecommendationsResponse)
async def recommend_recipes(
    keywords: str = Query(..., description="Comma-separated keywords"),
    target_calories: float = Query(..., gt=0, description="Target calorie amount"),
    top_n: Optional[int] = Query(5, ge=1, le=20, description="Number of results")
):
    try:
        keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
        if not keyword_list: raise HTTPException(400, "Keywords cannot be empty")
        results = search_food_by_keywords(data, keyword_list, target_calories, top_n)
        if results.empty: return {"recommendations": []}
        recommendations = []
        for _, row in results.iterrows():
            rec = RecipeRecommendation(
                RecipeId=int(row['RecipeId']), Name=row['food'], Calories=row.get('Calories'),
                ProteinContent=row.get('ProteinContent'), CarbohydrateContent=row.get('CarbohydrateContent'),
                FatContent=row.get('FatContent'),
                Image=complete_image_url(row['PrimaryImage']) if is_valid_image_url(row['PrimaryImage']) else None,
            )
            recommendations.append(rec)
        return {"recommendations": recommendations}
    except Exception as e:
        print(f"Error in /recommend_recipes: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.get("/recipe_detail/{recipe_id}", response_model=RecipeDetail)
async def get_recipe_detail(recipe_id: int):
    try:
        recipe = data[data['RecipeId'] == recipe_id]
        if recipe.empty: raise HTTPException(status_code=404, detail="Recipe not found")
        recipe = recipe.iloc[0]
        def parse_list_field(value):
            if isinstance(value, list): return value
            if pd.isna(value): return None
            try: return ast.literal_eval(value)
            except (ValueError, SyntaxError): return [item.strip() for item in str(value).split(',') if item.strip()] if value else None
        detail_image_url = parse_image_url(recipe.get('Images'))
        return RecipeDetail(
            RecipeId=int(recipe['RecipeId']), Name=recipe['Name'], CookTime=int(recipe.get('CookTime', 0)),
            PrepTime=int(recipe.get('PrepTime', 0)), TotalTime=int(recipe.get('TotalTime', 0)),
            Image=complete_image_url(detail_image_url) if detail_image_url else None,
            Keywords=parse_list_field(recipe.get('tags')), RecipeIngredientParts=parse_list_field(recipe.get('RecipeIngredientParts')),
            Calories=recipe.get('Calories'), FatContent=recipe.get('FatContent'), SaturatedFatContent=recipe.get('SaturatedFatContent'),
            CholesterolContent=recipe.get('CholesterolContent'), SodiumContent=recipe.get('SodiumContent'),
            CarbohydrateContent=recipe.get('CarbohydrateContent'), FiberContent=recipe.get('FiberContent'),
            SugarContent=recipe.get('SugarContent'), ProteinContent=recipe.get('ProteinContent')
        )
    except Exception as e:
        print(f"Error in /recipe_detail/{recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")