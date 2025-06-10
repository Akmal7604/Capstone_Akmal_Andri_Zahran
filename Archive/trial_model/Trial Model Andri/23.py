from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import re
import numpy as np
import json
import ast
import xgboost as xgb
from datetime import datetime

app = FastAPI()

# --- 1. Load Model and Preprocessing Objects ---
try:
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model('xgb_model.json')

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

    data = pd.read_csv('../recipes_new.csv')
    
    data.rename(columns={'Keywords': 'tags'}, inplace=True)
    data['tags'] = data['tags'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z, ]', '', x))
    data['tags'] = data['tags'].apply(lambda x: [tag.strip() for tag in x.lower().split(',') if tag.strip()])

    def is_valid_image_url(url: Optional[str]) -> bool:
        if not url or not isinstance(url, str):
            return False
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG', '.JPEG', '.PNG']
        return any(url.lower().endswith(ext) for ext in image_extensions)

    def complete_image_url(url: str) -> str:
        base_url = "https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/"
        if url.startswith("http"):
            return url
        return base_url + url.lstrip('/')

    def parse_image_url(image_str):
        if not image_str or not isinstance(image_str, str):
            return None
        try:
            image_str_clean = image_str.replace("'", '"')
            images = json.loads(image_str_clean)
            if isinstance(images, list) and images:
                for url in images:
                    if is_valid_image_url(url):
                        return url
        except json.JSONDecodeError:
            pass
        
        try:
            images = ast.literal_eval(image_str)
            if isinstance(images, list) and images:
                for url in images:
                    if is_valid_image_url(url):
                        return url
        except (ValueError, SyntaxError):
            pass
        
        for url in image_str.split(','):
            url = url.strip()
            if is_valid_image_url(url):
                return url
        return None

    data['PrimaryImage'] = data['Images'].apply(parse_image_url)

except Exception as e:
    print(f"Error during initialization: {e}")
    raise RuntimeError(f"Initialization failed: {e}")

# --- 2. Recommendation Function ---
def search_food_by_keywords(df, keywords, target_calories, top_n=5):
    keywords = [k.lower() for k in keywords]
    temp_df = df.copy()
    temp_df['tags_cleaned'] = temp_df['tags'].copy()

    filtered_df = temp_df[
        temp_df['tags_cleaned'].apply(lambda x: any(k in x for k in keywords))
    ].copy()

    empty_df_columns = ['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent',
                        'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage',
                        'DatePublished']

    if filtered_df.empty:
        return pd.DataFrame(columns=empty_df_columns)

    target_calorie_label = None
    if target_calories <= bins[0]:
        target_calorie_label = labels[0]
    elif target_calories >= bins[-1]:
        target_calorie_label = labels[-1]
    else:
        for i in range(len(bins) - 1):
            if bins[i] < target_calories <= bins[i + 1]:
                target_calorie_label = labels[i]
                break

    if target_calorie_label is None:
        return pd.DataFrame(columns=empty_df_columns)

    try:
        target_calorie_bin_encoded = loaded_le.transform([target_calorie_label])[0]
    except ValueError:
        return pd.DataFrame(columns=empty_df_columns)

    filtered_keywords_encoded = loaded_mlb.transform(filtered_df['tags_cleaned'])
    filtered_keywords_df = pd.DataFrame(filtered_keywords_encoded, columns=loaded_mlb.classes_, index=filtered_df.index)

    X_predict = pd.DataFrame(0, index=filtered_df.index, columns=loaded_model_input_feature_columns)

    numerical_features = [
        'CookTime', 'PrepTime', 'TotalTime', 'FatContent', 'SaturatedFatContent',
        'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent',
        'SugarContent', 'ProteinContent', 'RecipeServings'
    ]
    for col in numerical_features:
        if col in filtered_df.columns:
            X_predict[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

    for col in loaded_mlb.classes_:
        if col in filtered_keywords_df.columns:
            X_predict[col] = filtered_keywords_df[col]

    try:
        pred_proba = loaded_model.predict_proba(X_predict)
        filtered_df['pred_prob'] = pred_proba[:, target_calorie_bin_encoded]
    except Exception as e:
        print(f"Prediction error: {e}")
        return pd.DataFrame(columns=empty_df_columns)

    # --- Randomization Step ---
    filtered_df = filtered_df.sort_values(by='pred_prob', ascending=False)
    
    pool_size = min(len(filtered_df), max(top_n * 5, 50))
    
    relevant_pool = filtered_df.head(pool_size)

    if len(relevant_pool) <= top_n:
        final_results = relevant_pool
    else:
        final_results = relevant_pool.sample(n=top_n, random_state=None) 
    
    output_columns_map = {
        'RecipeId': 'RecipeId',
        'Name': 'Name',
        'tags': 'tags',
        'Calories': 'Calories',
        'pred_prob': 'pred_prob',
        'ProteinContent': 'ProteinContent',
        'CarbohydrateContent': 'CarbohydrateContent',
        'FatContent': 'FatContent',
        'RecipeServings': 'RecipeServings', # Keep this for internal use if needed, but not for ServingSize in API output
        'PrimaryImage': 'PrimaryImage',
        'DatePublished': 'DatePublished'
    }

    for original_col, _ in output_columns_map.items():
        if original_col not in final_results.columns:
            final_results[original_col] = np.nan

    final_results = final_results.rename(columns={'Name': 'food'})

    return final_results[list(output_columns_map.keys())].rename(columns={'Name': 'food'})

# --- 3. Pydantic Models ---
class RecipeRecommendation(BaseModel):
    RecipeId: int
    Name: str
    Calories: Optional[float] = None
    ProteinContent: Optional[float] = None
    CarbohydrateContent: Optional[float] = None
    FatContent: Optional[float] = None
    # This is where we ensure it's always 1
    ServingSize: int = 1 
    ServingUnit: Optional[str] = "Porsi"
    Image: Optional[str] = None

class RecommendationsResponse(BaseModel):
    recommendations: List[RecipeRecommendation]

class RecipeDetail(BaseModel):
    RecipeId: int
    Name: str
    CookTime: Optional[float] = None
    PrepTime: Optional[float] = None
    TotalTime: Optional[float] = None
    Image: Optional[str] = None
    Keywords: Optional[List[str]] = None
    RecipeIngredientParts: Optional[List[str]] = None
    Calories: Optional[float] = None
    FatContent: Optional[float] = None
    SaturatedFatContent: Optional[float] = None
    CholesterolContent: Optional[float] = None
    SodiumContent: Optional[float] = None
    CarbohydrateContent: Optional[float] = None
    FiberContent: Optional[float] = None
    SugarContent: Optional[float] = None
    ProteinContent: Optional[float] = None
    # This is where we ensure it's always 1
    ServingSize: int = 1 
    ServingUnit: Optional[str] = "Porsi"

# --- 4. API Endpoints ---
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

        recommendations = []
        for _, row in results.iterrows():
            rec = RecipeRecommendation(
                RecipeId=int(row['RecipeId']),
                Name=str(row['food']),
                Calories=float(row['Calories']) if pd.notna(row['Calories']) else None,
                ProteinContent=float(row['ProteinContent']) if pd.notna(row['ProteinContent']) else None,
                CarbohydrateContent=float(row['CarbohydrateContent']) if pd.notna(row['CarbohydrateContent']) else None,
                FatContent=float(row['FatContent']) if pd.notna(row['FatContent']) else None,
                # <<< REMOVED: ServingSize=row.get('RecipeServings', 1) >>>
                # By not providing 'ServingSize' here, Pydantic will use its default of 1.
                # ServingUnit also defaults.
                Image=complete_image_url(str(row['PrimaryImage'])) if pd.notna(row['PrimaryImage']) and is_valid_image_url(str(row['PrimaryImage'])) else None,
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
        if recipe.empty:
            raise HTTPException(status_code=404, detail="Recipe not found")
        recipe = recipe.iloc[0]

        def parse_list_field(value):
            if isinstance(value, list):
                return value
            if pd.isna(value):
                return None
            try:
                parsed = ast.literal_eval(str(value))
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            return [item.strip() for item in str(value).split(',') if item.strip()] if str(value).strip() else None

        detail_image_url = parse_image_url(recipe.get('Images'))
        
        return RecipeDetail(
            RecipeId=int(recipe['RecipeId']),
            Name=str(recipe['Name']),
            CookTime=float(recipe['CookTime']) if pd.notna(recipe['CookTime']) else None,
            PrepTime=float(recipe['PrepTime']) if pd.notna(recipe['PrepTime']) else None,
            TotalTime=float(recipe['TotalTime']) if pd.notna(recipe['TotalTime']) else None,
            Image=complete_image_url(str(detail_image_url)) if pd.notna(detail_image_url) and is_valid_image_url(str(detail_image_url)) else None,
            Keywords=parse_list_field(recipe.get('tags')),
            RecipeIngredientParts=parse_list_field(recipe.get('RecipeIngredientParts')),
            Calories=float(recipe['Calories']) if pd.notna(recipe['Calories']) else None,
            FatContent=float(recipe['FatContent']) if pd.notna(recipe['FatContent']) else None,
            SaturatedFatContent=float(recipe['SaturatedFatContent']) if pd.notna(recipe['SaturatedFatContent']) else None,
            CholesterolContent=float(recipe['CholesterolContent']) if pd.notna(recipe['CholesterolContent']) else None,
            SodiumContent=float(recipe['SodiumContent']) if pd.notna(recipe['SodiumContent']) else None,
            CarbohydrateContent=float(recipe['CarbohydrateContent']) if pd.notna(recipe['CarbohydrateContent']) else None,
            FiberContent=float(recipe['FiberContent']) if pd.notna(recipe['FiberContent']) else None,
            SugarContent=float(recipe['SugarContent']) if pd.notna(recipe['SugarContent']) else None,
            ProteinContent=float(recipe['ProteinContent']) if pd.notna(recipe['ProteinContent']) else None,
            # <<< REMOVED: ServingSize=recipe.get('RecipeServings', 1) >>>
            # By not providing 'ServingSize' here, Pydantic will use its default of 1.
            ServingUnit="Porsi"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in /recipe_detail/{recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")