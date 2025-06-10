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

    # Update the CSV path as per your local setup or change it to a relative path if deployed
    # For local development, 'C:/Users/andri/Documents/Andri Martin/Trisakti/Semester 6/DBS_Coding-Camp2025/Capstone/Capstone/Capstone_Akmal_Andri_Zahran/recipes_new.csv'
    # For deployment, consider a relative path like 'recipes_new.csv' in the same directory as main.py
    data = pd.read_csv('recipes_new.csv')
    
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
        if url.startswith("http"): # If it's already a full URL, return it
            return url
        return base_url + url.lstrip('/')

    def parse_image_url(image_str):
        if not image_str or not isinstance(image_str, str):
            return None
        try:
            # Try to parse as JSON first (handles '["url1", "url2"]')
            image_str_clean = image_str.replace("'", '"')
            images = json.loads(image_str_clean)
            if isinstance(images, list) and images:
                for url in images:
                    if is_valid_image_url(url):
                        return url
        except json.JSONDecodeError:
            pass # Fall through to next parsing attempt if JSON fails
        
        try:
            # Try to parse as literal eval (handles Python list-like strings '["url1", "url2"]')
            images = ast.literal_eval(image_str)
            if isinstance(images, list) and images:
                for url in images:
                    if is_valid_image_url(url):
                        return url
        except (ValueError, SyntaxError):
            pass # Fall through to simple split if literal_eval fails
        
        # Fallback to splitting by comma for simple comma-separated strings
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
def search_food_by_keywords(df, keywords, target_calories, top_n=5): # Removed seed parameter here
    keywords = [k.lower() for k in keywords]
    temp_df = df.copy()
    temp_df['tags_cleaned'] = temp_df['tags'].copy()

    filtered_df = temp_df[
        temp_df['tags_cleaned'].apply(lambda x: any(k in x for k in keywords))
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame(columns=['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent',
                                     'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage',
                                     'DatePublished'])

    # Tentukan label kalori sesuai bins dan labels
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

    if target_calorie_label is None: # Handle case where target_calories is out of defined bin range
        return pd.DataFrame(columns=['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent',
                                     'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage',
                                     'DatePublished'])


    try:
        target_calorie_bin_encoded = loaded_le.transform([target_calorie_label])[0]
    except ValueError:
        return pd.DataFrame(columns=['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent',
                                     'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage',
                                     'DatePublished'])

    # Encode tags menjadi fitur biner
    filtered_keywords_encoded = loaded_mlb.transform(filtered_df['tags_cleaned'])
    filtered_keywords_df = pd.DataFrame(filtered_keywords_encoded, columns=loaded_mlb.classes_, index=filtered_df.index)

    # Prepare fitur input model
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
        return pd.DataFrame(columns=['RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent',
                                     'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage',
                                     'DatePublished'])

    # --- Randomization Step ---
    # We will sort by prediction probability first, then introduce randomness
    # by taking a larger sample and then randomly selecting from that.
    
    # Sort by probability descending to prioritize more relevant items
    filtered_df = filtered_df.sort_values(by='pred_prob', ascending=False)
    
    # Define a pool size. We take a pool of recipes that are somewhat relevant
    # (e.g., top 100 or all if less than 100) before random sampling.
    # This prevents sampling from very low-probability recipes if there are many matches.
    pool_size = min(len(filtered_df), max(top_n * 5, 50)) # Example: get top 50 or 5 times top_n
    
    # Take a relevant pool of recipes
    relevant_pool = filtered_df.head(pool_size)

    # If the relevant pool is smaller than top_n, just take all available from the pool
    if len(relevant_pool) <= top_n:
        final_results = relevant_pool
    else:
        # Randomly sample 'top_n' recipes from this relevant pool
        # random_state=None ensures a different sample each time
        final_results = relevant_pool.sample(n=top_n, random_state=None) 
    
    # --- Ensure all required columns are present ---
    # Define a list of all columns expected in the final output
    output_columns = [
        'RecipeId', 'Name', 'tags', 'Calories', 'pred_prob', 'ProteinContent',
        'CarbohydrateContent', 'FatContent', 'RecipeServings', 'PrimaryImage',
        'DatePublished'
    ]
    # Ensure all columns exist, filling missing ones with NaN if needed
    for col in output_columns:
        if col not in final_results.columns:
            final_results[col] = np.nan # Or appropriate default based on type

    return final_results[output_columns].rename(columns={'Name': 'food'})

# --- 3. Pydantic Models ---
class RecipeRecommendation(BaseModel):
    RecipeId: int
    Name: str
    Calories: float
    ProteinContent: float
    CarbohydrateContent: float
    FatContent: float
    ServingSize: int = 1
    ServingUnit: str = "Porsi"
    Image: Optional[str]

class RecommendationsResponse(BaseModel):
    recommendations: List[RecipeRecommendation]

class RecipeDetail(BaseModel):
    RecipeId: int
    Name: str
    CookTime: int
    PrepTime: int
    TotalTime: int
    Image: Optional[str]
    Keywords: List[str]
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    ServingSize: int = 1
    ServingUnit: str = "Porsi"

# --- 4. API Endpoints ---
@app.get("/recommend_recipes", response_model=RecommendationsResponse)
async def recommend_recipes(
    keywords: str = Query(..., description="Comma-separated keywords (e.g., 'chicken,easy,dinner')"),
    target_calories: float = Query(..., gt=0, description="Target calorie amount"),
    top_n: Optional[int] = Query(5, ge=1, le=20, description="Number of results")
    # Removed 'seed' from API endpoint as it's no longer needed for per-call randomness
):
    try:
        keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
        if not keyword_list:
            raise HTTPException(400, "Keywords cannot be empty")

        # Call search_food_by_keywords without the seed
        results = search_food_by_keywords(data, keyword_list, target_calories, top_n)
        
        if results.empty:
            return {"recommendations": []}

        recommendations = []
        for _, row in results.iterrows():
            # Ensure 'tags' are processed to list of strings
            tags_list = [str(tag) for tag in row['tags']] if isinstance(row['tags'], list) else []

            rec = RecipeRecommendation(
                RecipeId=row['RecipeId'],
                Name=row['food'], # Use 'food' as per the rename in search_food_by_keywords
                Calories=row.get('Calories'),
                ProteinContent=row.get('ProteinContent'),
                CarbohydrateContent=row.get('CarbohydrateContent'),
                FatContent=row.get('FatContent'),
                Image=complete_image_url(row['PrimaryImage']) if is_valid_image_url(row['PrimaryImage']) else None,
                # Removed 'tags' here if it's not part of the Recommendation model,
                # or add it if you want it back in RecipeRecommendation
                # tags=tags_list # Uncomment if you add tags back to RecipeRecommendation model
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

        # Helper function to parse list-like fields from strings
        def parse_list_field(value):
            if isinstance(value, list):
                return value
            if pd.isna(value):
                return None # Return None for NaN values
            try:
                # Try to parse as a Python literal (e.g., '["item1", "item2"]')
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            # Fallback for simple comma-separated strings
            return [item.strip() for item in str(value).split(',') if item.strip()] if value else None

        # Apply image parsing for the detail view as well
        detail_image_url = parse_image_url(recipe.get('Images'))
        
        return RecipeDetail(
            RecipeId=recipe['RecipeId'],
            Name=recipe['Name'],
            CookTime=recipe.get('CookTime'),
            PrepTime=recipe.get('PrepTime'),
            TotalTime=recipe.get('TotalTime'),
            Image=complete_image_url(detail_image_url) if detail_image_url else None, # Use the parsed and completed URL
            Keywords=parse_list_field(recipe.get('tags')), # Use 'tags' from renamed column
            RecipeIngredientParts=parse_list_field(recipe.get('RecipeIngredientParts')),
            Calories=recipe.get('Calories'),
            FatContent=recipe.get('FatContent'),
            SaturatedFatContent=recipe.get('SaturatedFatContent'),
            CholesterolContent=recipe.get('CholesterolContent'),
            SodiumContent=recipe.get('SodiumContent'),
            CarbohydrateContent=recipe.get('CarbohydrateContent'),
            FiberContent=recipe.get('FiberContent'),
            SugarContent=recipe.get('SugarContent'),
            ProteinContent=recipe.get('ProteinContent')
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in /recipe_detail/{recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")