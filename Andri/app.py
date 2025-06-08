# app.py (FastAPI version)
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import re
import numpy as np

# --- 1. Load the necessary pre-trained objects and data ---
try:
    # ... (pemuatan model pickle lainnya tetap sama) ...
    loaded_model = pickle.load(open('C:/Users/andri/Documents/Andri Martin/Trisakti/Semester 6/DBS_Coding-Camp2025/Capstone/Capstone/Capstone_Akmal_Andri_Zahran/Andri/xgb_model.pkl', 'rb'))
    with open('C:/Users/andri/Documents/Andri Martin/Trisakti/Semester 6/DBS_Coding-Camp2025/Capstone/Capstone/Capstone_Akmal_Andri_Zahran/Andri/mlb.pkl', 'rb') as f:
        loaded_mlb = pickle.load(f)
    with open('C:/Users/andri/Documents/Andri Martin/Trisakti/Semester 6/DBS_Coding-Camp2025/Capstone/Capstone/Capstone_Akmal_Andri_Zahran/Andri/le.pkl', 'rb') as f:
        loaded_le = pickle.load(f)
    with open('C:/Users/andri/Documents/Andri Martin/Trisakti/Semester 6/DBS_Coding-Camp2025/Capstone/Capstone/Capstone_Akmal_Andri_Zahran/Andri/model_input_feature_columns.pkl', 'rb') as f:
        loaded_model_input_feature_columns = pickle.load(f)

    data = pd.read_csv('C:/Users/andri/Documents/Andri Martin/Trisakti/Semester 6/DBS_Coding-Camp2025/Capstone/Capstone/Capstone_Akmal_Andri_Zahran/recipes_new.csv')
    columns = [
        'RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime',
        'RecipeIngredientParts', 'Calories', 'FatContent', 'SaturatedFatContent',
        'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
        'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeServings', 'Keywords',
        'Images'
    ]
    data = data[columns]
    data.rename(columns={'Keywords': 'tags'}, inplace=True)

    # --- MODIFIKASI PEMROSESAN TAG DIMULAI DI SINI ---
    # 1. Ganti tanda hubung (-) dengan spasi. Ini penting AGAR "high-protein" menjadi "high protein".
    data['tags'] = data['tags'].astype(str).apply(lambda x: x.replace('-', ' '))
    
    # 2. Hapus karakter yang tidak diinginkan (selain huruf, angka, koma, dan spasi).
    #    Regex diubah sedikit untuk memperbolehkan angka jika memang ada dalam tag (misal "60 minute recipes").
    #    Jika Anda yakin hanya huruf, koma, dan spasi yang diinginkan, Anda bisa kembali ke r'[^a-zA-Z, ]'
    data['tags'] = data['tags'].apply(lambda x: re.sub(r'[^a-zA-Z0-9, ]', '', x)) # Memperbolehkan angka
    # data['tags'] = data['tags'].apply(lambda x: re.sub(r'[^a-zA-Z, ]', '', x)) # Versi lama jika angka tidak dibutuhkan

    # 3. Ubah ke huruf kecil, pisahkan dengan koma, hilangkan spasi ekstra, dan buang tag kosong.
    data['tags'] = data['tags'].apply(
        lambda x: [tag.strip() for tag in x.lower().split(',') if tag.strip()]
    )
    # --- MODIFIKASI PEMROSESAN TAG SELESAI ---

    bins = [0, 200, 400, 600, 800, 1000, np.inf]
    labels = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1000+']

    print("All necessary objects and data loaded successfully for FastAPI app.")
    # print(data[['Name', 'tags']].head()) # Untuk debugging, lihat hasil pemrosesan tags

except Exception as e:
    print(f"Error loading required files: {e}. Please ensure 'xgb_model.pkl', 'mlb.pkl', 'le.pkl', 'model_input_feature_columns.pkl' and 'recipes_new.csv' are in the correct directories.")
    raise RuntimeError(f"Failed to load essential files: {e}")

# --- 2. Define the search_food_by_keywords function ---
# (Tidak ada perubahan di fungsi ini, karena masalahnya ada di data 'tags' yang disiapkan)
def search_food_by_keywords(df, keywords: List[str], target_calories: float, top_n: int = 5):
    keywords = [k.lower() for k in keywords]
    temp_df = df.copy()
    
    # Logika filter yang diperbaiki untuk mendukung 1 keyword
    temp_df['tags_cleaned'] = temp_df['tags'] 
    filtered_df = temp_df[temp_df['tags_cleaned'].apply(
        lambda tags: any(kw in ' '.join(tags) for kw in keywords) if isinstance(tags, list) else False
    )].copy()

    if filtered_df.empty:
        return pd.DataFrame(columns=['food', 'tags', 'calories', 'pred_prob'])

    target_calorie_label = None
    if target_calories <= bins[0]:
        target_calorie_label = labels[0]
    elif target_calories > bins[-2]: 
        target_calorie_label = labels[-1]
    else:
        for i in range(len(bins) - 1):
            if bins[i] < target_calories <= bins[i+1]:
                target_calorie_label = labels[i]
                break
    
    if target_calorie_label is None:
        # Fallback jika tidak ada label yang cocok, bisa jadi karena nilai kalori ekstrem
        # Anda bisa menambahkan logika default di sini jika diperlukan
        print(f"Warning: Could not determine target calorie label for {target_calories}. Using fallback or expecting error.")
        # Untuk contoh, jika target_calories sangat tinggi, kita bisa set ke label terakhir
        if target_calories > bins[-2]: target_calorie_label = labels[-1]
        # Jika sangat rendah (misalnya 0 atau negatif, meskipun validasi Query(gt=0) harusnya mencegah ini)
        elif target_calories <= bins[0]: target_calorie_label = labels[0]
        else: # Jika masih null, ini kondisi tak terduga
             print(f"Error: target_calorie_label is None for target_calories: {target_calories}. Cannot proceed without a label.")
             return pd.DataFrame(columns=['food', 'tags', 'calories', 'pred_prob'])


    try:
        target_calorie_bin_encoded = loaded_le.transform([target_calorie_label])[0]
    except ValueError:
        print(f"Error: Target calorie label '{target_calorie_label}' (from target_calories: {target_calories}) not found in loaded LabelEncoder classes: {loaded_le.classes_}")
        return pd.DataFrame(columns=['food', 'tags', 'calories', 'pred_prob'])
    except TypeError: 
        print(f"Error: target_calorie_label is None for target_calories: {target_calories}. Cannot transform.")
        return pd.DataFrame(columns=['food', 'tags', 'calories', 'pred_prob'])

    # Pastikan 'tags_cleaned' yang akan di-transform oleh MLB adalah list of lists (atau iterable of iterables)
    # Jika 'tags_cleaned' adalah Series of lists, ini sudah benar.
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

    for col in loaded_mlb.classes_: # Ini adalah semua kemungkinan tag dari training
        if col in filtered_keywords_df.columns: # Jika tag ini ada di resep yang difilter
            X_predict[col] = filtered_keywords_df[col]
        # else: X_predict[col] akan tetap 0 (default dari inisialisasi DataFrame)

    # Pastikan urutan kolom X_predict sesuai dengan yang diharapkan model
    X_predict = X_predict[loaded_model_input_feature_columns]

    pred_proba = loaded_model.predict_proba(X_predict)

    if target_calorie_bin_encoded >= pred_proba.shape[1]:
        print(f"Error: Encoded target calorie bin {target_calorie_bin_encoded} (label: {target_calorie_label}) is out of bounds for model's prediction probabilities (num_classes={pred_proba.shape[1]}).")
        return pd.DataFrame(columns=['food', 'tags', 'calories', 'pred_prob'])

    filtered_df['pred_prob'] = [prob[target_calorie_bin_encoded] for prob in pred_proba]
    filtered_df = filtered_df.sort_values(by='pred_prob', ascending=False)

    return filtered_df[['Name', 'tags', 'Calories', 'pred_prob']].rename(
        columns={'Name': 'food'}
    ).head(top_n)

# --- 3. Initialize FastAPI App ---
app = FastAPI()

# --- Define Pydantic Models for Response ---
class Recommendation(BaseModel):
    food: str
    tags: List[str]
    Calories: float
    pred_prob: float

class RecommendationsResponse(BaseModel):
    recommendations: List[Recommendation]

# --- 4. Define API Endpoint with FastAPI Decorators ---
# (Tidak ada perubahan di endpoint ini)
@app.get("/recommend_recipes", response_model=RecommendationsResponse)
async def recommend_recipes(
    keywords: str = Query(..., description="Comma-separated list of keywords (e.g., 'chicken,easy,dinner')"),
    target_calories: float = Query(..., gt=0, description="Target calorie amount (e.g., 500)"),
    top_n: Optional[int] = Query(5, ge=1, le=20, description="Number of recommendations to return")
):
    """
    Recommends recipes based on keywords (comma-separated string) and target calorie range.
    Example: /recommend_recipes?keywords=chicken,pasta&target_calories=600&top_n=3
    """
    try:
        keyword_list = [keyword.strip().lower() for keyword in keywords.split(',') if keyword.strip()]
        if not keyword_list:
            raise HTTPException(status_code=400, detail="Keywords cannot be empty.")

        recommendations_df = search_food_by_keywords(
            data, keyword_list, target_calories, top_n
        )

        if recommendations_df.empty:
            return {"recommendations": []}
        else:
            return {"recommendations": recommendations_df.to_dict(orient='records')}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- 5. Run the FastAPI App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)