# --- 1. Imports ---
import pandas as pd
import uvicorn
import ast
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- 2. Data Simulation (Ganti bagian ini dengan memuat file CSV Anda) ---
# Karena file recipes_new.csv tidak tersedia, kita buat data tiruan.
# Anda harus mengganti bagian ini dengan: data = pd.read_csv('recipes_new.csv')
data = pd.read_csv('../../../Capstone_Kalkulori/recipes_new.csv')


# --- 3. Helper Functions (Implementasi sederhana agar kode berjalan) ---

def search_food_by_keywords(df, keywords, target_calories, top_n):
    """Fungsi pencarian berdasarkan keyword dan kalori terdekat."""
    # Pastikan tags adalah list
    df['tags_list'] = df['tags'].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x))
    
    # Filter berdasarkan keywords
    keyword_set = set(k.lower() for k in keywords)
    mask = df['tags_list'].apply(lambda tags: keyword_set.issubset(set(t.lower() for t in tags)))
    filtered_df = df[mask].copy()

    # Hitung selisih kalori dan urutkan
    filtered_df['calorie_diff'] = (filtered_df['Calories'] - target_calories).abs()
    results = filtered_df.sort_values('calorie_diff').head(top_n)
    return results

def is_valid_image_url(url):
    """Placeholder untuk validasi URL gambar."""
    return isinstance(url, str) and url.startswith('http')

def complete_image_url(url):
    """Placeholder untuk melengkapi URL gambar jika perlu."""
    return url


# --- 4. Pydantic Models ---
class RecipeRecommendation(BaseModel):
    RecipeId: int
    Name: str
    Calories: float
    ProteinContent: float
    CarbohydrateContent: float
    FatContent: float
    ServingSize: int = 1
    ServingUnit: str = "Porsi"
    Image: Optional[str] = None

class RecommendationsResponse(BaseModel):
    recommendations: List[RecipeRecommendation]

class RecipeDetail(BaseModel):
    RecipeId: int
    Name: str
    CookTime: int
    PrepTime: int
    TotalTime: int
    Image: Optional[str] = None
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

# --- 5. FastAPI App Initialization ---
app = FastAPI(
    title="Food Recipe API",
    description="API untuk mencari dan merekomendasikan resep makanan.",
    version="1.0.0"
)

# --- 6. API Endpoints ---
@app.get("/search", response_model=RecommendationsResponse)
async def search_recipes_by_name(
    name: str = Query(..., min_length=1, description="Potongan atau nama lengkap makanan yang ingin dicari"),
    top_n: int = Query(10, ge=1, le=50, description="Jumlah maksimal hasil yang ditampilkan")
):
    """
    Mencari resep berdasarkan nama. Pencarian ini bersifat case-insensitive
    dan akan mencocokkan resep meskipun query hanya sebagian dari nama lengkap.
    """
    try:
        search_results = data[data['Name'].str.contains(name, case=False, na=False)].head(top_n)

        if search_results.empty:
            return {"recommendations": []}

        recommendations = []
        for _, row in search_results.iterrows():
            rec = RecipeRecommendation(
                RecipeId=row['RecipeId'],
                Name=row['Name'],
                Calories=row['Calories'],
                ProteinContent=row['ProteinContent'],
                CarbohydrateContent=row['CarbohydrateContent'],
                FatContent=row['FatContent'],
                Image=complete_image_url(row['Images']) if is_valid_image_url(row['Images']) else None,
            )
            recommendations.append(rec)

        return {"recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")

@app.get("/recommend_recipes", response_model=RecommendationsResponse)
async def recommend_recipes(
    keywords: str = Query(..., description="Comma-separated keywords (e.g., 'chicken,dinner')"),
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
                RecipeId=row['RecipeId'],
                Name=row['Name'],
                Calories=row['Calories'],
                ProteinContent=row['ProteinContent'],
                CarbohydrateContent=row['CarbohydrateContent'],
                FatContent=row['FatContent'],
                Image=complete_image_url(row['PrimaryImage']) if is_valid_image_url(row['PrimaryImage']) else None,
            )
            recommendations.append(rec)

        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.get("/recipe_detail/{recipe_id}", response_model=RecipeDetail)
async def get_recipe_detail(recipe_id: int):
    try:
        recipe_df = data[data['RecipeId'] == recipe_id]
        if recipe_df.empty:
            raise HTTPException(status_code=404, detail="Recipe not found")
        recipe = recipe_df.iloc[0]

        def parse_list_field(value):
            if isinstance(value, list):
                return value
            try:
                # Untuk kasus data dari CSV dibaca sebagai string
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [str(value)] if value else []

        return RecipeDetail(
            RecipeId=int(recipe['RecipeId']),
            Name=recipe['Name'],
            CookTime=int(recipe['CookTime']),
            PrepTime=int(recipe['PrepTime']),
            TotalTime=int(recipe['TotalTime']),
            Image=recipe['Images'],
            Keywords=parse_list_field(recipe['tags']),
            RecipeIngredientParts=parse_list_field(recipe['RecipeIngredientParts']),
            Calories=float(recipe['Calories']),
            FatContent=float(recipe['FatContent']),
            SaturatedFatContent=float(recipe['SaturatedFatContent']),
            CholesterolContent=float(recipe['CholesterolContent']),
            SodiumContent=float(recipe['SodiumContent']),
            CarbohydrateContent=float(recipe['CarbohydrateContent']),
            FiberContent=float(recipe['FiberContent']),
            SugarContent=float(recipe['SugarContent']),
            ProteinContent=float(recipe['ProteinContent'])
        )
    except Exception as e:
        raise HTTPException(500, f"Internal error: {str(e)}")


# --- 7. Run the App ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)