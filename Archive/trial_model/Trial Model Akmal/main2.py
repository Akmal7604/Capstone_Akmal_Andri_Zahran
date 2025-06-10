import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
from typing import List, Union, Optional, Dict

# --- Bagian 1: Definisi Kelas MealPlanGenerator ---

# --- Bagian 1: Definisi Kelas MealPlanGenerator ---

class MealPlanGenerator:
    def __init__(self, df, preloaded_scaler=None, preloaded_kmeans=None):
        self.df = df.copy()
        self.keyword_columns = []
        # Logika prioritas baru diintegrasikan di sini
        self.preprocess_data()

        if preloaded_scaler and preloaded_kmeans:
            self.scaler = preloaded_scaler
            self.kmeans = preloaded_kmeans
            self._assign_clusters_from_loaded_model()
        else:
            self.scaler = StandardScaler()
            self.kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
            self._cluster_recipes_and_fit()

    def preprocess_data(self):
        def parse_keywords(keywords):
            try:
                if isinstance(keywords, str):
                    # Konsisten menggunakan lowercase untuk pencocokan
                    return [k.strip().lower() for k in keywords.split(',') if k.strip()]
                elif isinstance(keywords, list):
                    return [str(k).strip().lower() for k in keywords if str(k).strip()]
                return []
            except Exception:
                return []

        ## << KAMUS BARU ANDA YANG DIINTEGRASIKAN KE DALAM STRUKTUR PRIORITAS >>
        # Kamus ini menerjemahkan daftar Anda ke dalam struktur 'specific' dan 'general'
        prioritized_keywords = {
            'Breakfast': {
                'specific': [
                    'breakfast', 'brunch', 'oatmeal', 'scones'
                ],
                'general': [
                    'breads', 'yeast breads', 'quick breads', 'sourdough breads', 'bread machine',
                    'smoothies', 'shakes', 'fruit', 'berries', 'strawberry', 'raspberries',
                    'cherries', 'apple', 'mango', 'pineapple', 'kiwifruit', 'plums', 'pears',
                    'melons', 'papaya', 'tropical fruits', 'oranges', 'grapes', 'citrus',
                    'peanut butter', 'egg free'
                ]
            },
            'Lunch': {
                'specific': [
                    'lunch/snacks', 'lunch', 'salad', 'sandwich', 'salad dressings',
                    'clear soup', 'mushroom soup', 'beef barley soup', 'chowders', 'gumbo',
                    'spreads', 'summer dip'
                ],
                'general': [
                    'weeknight', '< 15 mins', '< 30 mins', '< 60 mins', 'no cook',
                    'kid friendly', 'toddler friendly', 'college food', 'inexpensive',
                    'greens', 'spinach', 'collard greens', 'chard', 'cauliflower', 'artichoke',
                    'potato', 'yam/sweet potato', 'corn', 'peppers',
                    'beans', 'black beans', 'lentil', 'soy/tofu', 'tempeh',
                    'cheese', 'avocado', 'coconut', 'nuts'
                ]
            },
            'Dinner': {
                'specific': [
                    'dinner', 'stir fry', 'roast', 'stew', 'pot pie', 'meatloaf', 'meatballs',
                    'one dish meal', 'for large groups', 'potluck',
                    'meat', 'chicken', 'fish', 'steak', 'poultry', 'pork', 'lamb/sheep',
                    'veal', 'duck', 'goose', 'wild game', 'rabbit', 'deer', 'elk', 'moose',
                    'tuna', 'trout', 'halibut', 'bass', 'perch', 'catfish', 'mahi mahi',
                    'whitefish', 'orange roughy', 'tilapia', 'crab', 'lobster', 'crawfish',
                    'mussels', 'oysters', 'octopus', 'squid', 'quail', 'pheasant',
                    'whole chicken', 'chicken breast', 'chicken thigh & leg', 'chicken livers',
                    'whole duck', 'duck breasts', 'whole turkey', 'turkey breasts',
                    'beef organ meats', 'beef liver', 'beef kidney', 'roast beef', 'ham',
                    'pasta', 'penne', 'spaghetti', 'macaroni and cheese', 'manicotti', 'pasta shells', 'pasta elbow',
                    'rice', 'grains', 'brown rice', 'white rice', 'long grain rice', 'medium grain rice', 'short grain rice',
                    'mashed potatoes', 'indonesian', 'chinese', 'japanese', 'korean', 'vietnamese', 'thai', 'filipino',
                    'malaysian', 'cantonese', 'szechuan', 'hunan', 'mongolian', 'cambodian', 'nepalese', 'asian',
                    'indian', 'pakistani', 'lebanese', 'turkish', 'iraqi', 'southwest asia (middle east)', 'palestinian',
                    'mexican', 'caribbean', 'cuban', 'puerto rican', 'costa rican', 'south american', 'brazilian',
                    'peruvian', 'colombian', 'chilean', 'venezuelan', 'ecuadorean', 'guatemalan', 'honduran',
                    'greek', 'spanish', 'portuguese', 'german', 'belgian', 'dutch', 'swiss', 'austrian',
                    'hungarian', 'polish', 'czech', 'russian', 'european', 'scandinavian', 'swedish',
                    'norwegian', 'finnish', 'danish', 'icelandic', 'african', 'nigerian', 'moroccan',
                    'egyptian', 'ethiopian', 'south african', 'somalian', 'sudanese', 'cajun', 'creole',
                    'tex mex', 'hawaiian', 'southwestern u.s.', 'native american', 'canadian', 'australian',
                    'new zealand', 'polynesian', 'georgian', 'welsh', 'scottish'
                ],
                'general': [
                    'potato', 'beans', 'cheese', 'greens', 'corn',
                    'oven', 'broil/grill', 'stove top', 'pressure cooker', 'deep fried', 'baking'
                ]
            }
        }

        self.df['ParsedKeywords'] = self.df['Keywords'].apply(parse_keywords)

        def assign_type_and_priority(parsed_keywords_list):
            if not isinstance(parsed_keywords_list, list):
                return 'Other', 0
            
            for meal_type, priorities in prioritized_keywords.items():
                if any(kw in parsed_keywords_list for kw in priorities['specific']):
                    return meal_type, 3 # Prioritas Tinggi
                if any(kw in parsed_keywords_list for kw in priorities['general']):
                    return meal_type, 1 # Prioritas Rendah
            
            return 'Other', 0

        self.df[['MealType', 'MealPriority']] = self.df['ParsedKeywords'].apply(
            lambda x: pd.Series(assign_type_and_priority(x))
        )
        
        valid_keywords = self.df['ParsedKeywords'].apply(lambda x: [str(item) for item in x] if isinstance(x, list) else [])
        mlb = MultiLabelBinarizer()
        mlb.fit_transform(valid_keywords) 
        self.keyword_columns = mlb.classes_

    def _cluster_recipes_and_fit(self):
        features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        if features.isnull().values.any():
            self.df.dropna(subset=['Calories', 'ProteinContent', 'CarbohydrateContent'], inplace=True)
            features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        features_scaled = self.scaler.fit_transform(features)
        self.df['Cluster'] = self.kmeans.fit_predict(features_scaled)

    def _assign_clusters_from_loaded_model(self):
        features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        if features.isnull().values.any():
            self.df.dropna(subset=['Calories', 'ProteinContent', 'CarbohydrateContent'], inplace=True)
            features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        features_scaled = self.scaler.transform(features)
        self.df['Cluster'] = self.kmeans.predict(features_scaled)

    def generate_meal_plans(self, total_calories_target, max_plans=4, calorie_tolerance_percent=0.10):
        meal_plans = []

        # Langkah 1: Filter DataFrame dasar
        breakfast_df = self.df[(self.df['MealType'] == 'Breakfast') & (self.df['MealPriority'] > 0)].copy()
        lunch_df = self.df[(self.df['MealType'] == 'Lunch') & (self.df['MealPriority'] > 0)].copy()
        dinner_df = self.df[(self.df['MealType'] == 'Dinner') & (self.df['MealPriority'] > 0)].copy()

        # Validasi awal yang lebih kuat
        if breakfast_df.empty or lunch_df.empty or dinner_df.empty:
            print("Peringatan Kritis: Salah satu kategori makanan (Sarapan/Makan Siang/Makan Malam) tidak memiliki resep yang valid setelah pemfilteran prioritas.")
            return [] # Langsung keluar

        # Langkah 2: Pembersihan dan persiapan data bobot (weights) secara eksplisit
        # Ini adalah langkah defensif untuk mencegah error tipe data.
        for df in [breakfast_df, lunch_df, dinner_df]:
            df['MealPriority'] = pd.to_numeric(df['MealPriority'], errors='coerce').fillna(0)

        additional_meal_pool = pd.concat([breakfast_df, lunch_df, dinner_df]).reset_index(drop=True)
        additional_meal_pool['MealPriority'] = pd.to_numeric(additional_meal_pool['MealPriority'], errors='coerce').fillna(0)

        calorie_threshold = 2300
        num_additional_meals = 2 if total_calories_target > calorie_threshold else 0
        
        attempts = 0
        max_attempts = max(max_plans * 250, 3000)
        generated_plan_signatures = set()

        while len(meal_plans) < max_plans and attempts < max_attempts:
            attempts += 1
            try:
                # Inisialisasi variabel resep
                breakfast, lunch, dinner = None, None, None
                
                # Sampling dengan validasi bobot yang ketat
                if not breakfast_df.empty and breakfast_df['MealPriority'].sum() > 0:
                    breakfast = breakfast_df.sample(1, weights=breakfast_df['MealPriority']).iloc[0]
                
                if not lunch_df.empty and lunch_df['MealPriority'].sum() > 0:
                    lunch = lunch_df.sample(1, weights=lunch_df['MealPriority']).iloc[0]

                if not dinner_df.empty and dinner_df['MealPriority'].sum() > 0:
                    dinner = dinner_df.sample(1, weights=dinner_df['MealPriority']).iloc[0]

                # Jika salah satu gagal di-sample, lewati iterasi ini
                if breakfast is None or lunch is None or dinner is None:
                    continue

                plan_meals = [
                    {'MealType': 'Breakfast', 'Recipe': breakfast},
                    {'MealType': 'Lunch', 'Recipe': lunch},
                    {'MealType': 'Dinner', 'Recipe': dinner}
                ]
                
                if num_additional_meals > 0:
                    if not additional_meal_pool.empty and additional_meal_pool['MealPriority'].sum() > 0:
                        additional_meals_df_sampled = additional_meal_pool.sample(
                            num_additional_meals, 
                            weights=additional_meal_pool['MealPriority']
                        )
                        for i, (index, meal_series) in enumerate(additional_meals_df_sampled.iterrows()):
                             plan_meals.append({'MealType': f'Additional Meal {i+1}', 'Recipe': meal_series})

            # Langkah 3: Penanganan Error yang Lebih Luas dan Informatif
            # Kita menangkap 'Exception' (semua jenis error), bukan hanya 'ValueError'.
            except Exception as e:
                # Ini akan MENCETAK error yang sebenarnya di konsol server Anda!
                print(f"!!! Terjadi Error saat sampling: {e} | Tipe Error: {type(e)} !!!")
                continue 

            total_plan_calories = sum(meal['Recipe']['Calories'] for meal in plan_meals)
            lower_bound = total_calories_target * (1 - calorie_tolerance_percent)
            
            if lower_bound <= total_plan_calories <= total_calories_target:
                plan_signature = tuple(sorted([meal['Recipe'].loc['RecipeId'] for meal in plan_meals]))
                if plan_signature in generated_plan_signatures:
                    continue

                meal_plan_details = []
                for meal in plan_meals:
                    recipe = meal['Recipe']
                    meal_plan_details.append({
                        'MealType': meal['MealType'],
                        'RecipeId': int(recipe.loc['RecipeId']),
                        'Name': recipe.loc['Name'],
                        'Calories': float(recipe.loc['Calories'])
                    })

                meal_plans.append({
                    'Meals': meal_plan_details,
                    'TotalCalories': float(total_plan_calories)
                })
                generated_plan_signatures.add(plan_signature)
        
        return meal_plans
    
    def save_models(self, kmeans_path="kmeans_model.joblib", scaler_path="scaler.joblib"):
        if self.kmeans is not None: joblib.dump(self.kmeans, kmeans_path)
        if self.scaler is not None: joblib.dump(self.scaler, scaler_path)

# --- Bagian 2: Aplikasi FastAPI ---
app = FastAPI(
    title="Meal Plan Generator API",
    description="API untuk menghasilkan rencana makan berdasarkan target kalori.",
    version="1.2.0" # Versi diperbarui
)

generator_instance: Optional[MealPlanGenerator] = None
data_loaded_successfully = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(BASE_DIR, "kmeans_model.joblib")
CSV_PATH = "D:\\UB\\Dicoding\\GitHub\\Capstone_Akmal_Andri_Zahran\\recipes_new.csv"


@app.on_event("startup")
async def load_resources():
    global generator_instance, data_loaded_successfully
    try:
        print(f"Mencoba memuat CSV dari: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)

        required_cols = ['RecipeId', 'Name', 'Keywords', 'Calories', 'ProteinContent', 'CarbohydrateContent']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise FileNotFoundError(f"Kolom yang dibutuhkan tidak ada di CSV: {missing}")

        numeric_cols = ['Calories', 'ProteinContent', 'CarbohydrateContent']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Keywords'] = df['Keywords'].fillna('')
        df.dropna(subset=numeric_cols, inplace=True)
        if df.empty:
            raise ValueError("DataFrame kosong setelah membersihkan NaN.")
        
        print("Data CSV berhasil dimuat dan dibersihkan.")

        try:
            print(f"Mencoba memuat model dari: {SCALER_PATH} dan {KMEANS_PATH}")
            loaded_scaler = joblib.load(SCALER_PATH)
            loaded_kmeans = joblib.load(KMEANS_PATH)
            print("✔️ Model berhasil dimuat dari file.")
            generator_instance = MealPlanGenerator(
                df, 
                preloaded_scaler=loaded_scaler, 
                preloaded_kmeans=loaded_kmeans
            )
        except FileNotFoundError:
            print("⚠️ Peringatan: File model tidak ditemukan. Memulai proses training baru...")
            generator_instance = MealPlanGenerator(df)
            print("✅ Training selesai. Menyimpan model baru...")
            generator_instance.save_models(kmeans_path=KMEANS_PATH, scaler_path=SCALER_PATH)
            print(f"✔️ Model baru berhasil disimpan di {BASE_DIR}")

        data_loaded_successfully = True
        print("🚀 Sumber daya berhasil dimuat. MealPlanGenerator siap digunakan.")
    except Exception as e:
        print(f"❌ Error kritis saat startup: {e}")
        data_loaded_successfully = False


# Model Pydantic tetap sama, karena struktur fleksibel ini masih sempurna
class MealDetail(BaseModel):
    MealType: str
    RecipeId: int
    Name: str
    Calories: float

class MealPlanResponse(BaseModel):
    Meals: List[MealDetail]
    TotalCalories: float

class MealPlansListResponse(BaseModel):
    meal_plans: List[MealPlanResponse]
    message: Optional[str] = None


@app.get(
    "/generate-meal-plan/",
    response_model=MealPlansListResponse,
    summary="Generate Meal Plans",
    ## << DESKRIPSI DIPERBARUI >>
    description="Menghasilkan beberapa rencana makan berdasarkan target kalori harian. Jika target > 2300 kalori, akan menghasilkan rencana 5 makanan (3 utama + 2 tambahan)."
)
async def generate_plans_endpoint(
    total_calories: float = Query(..., gt=0, description="Target total kalori harian (harus lebih dari 0)"),
    max_plans: int = Query(3, ge=1, le=10, description="Jumlah maksimum rencana makan yang akan dihasilkan"),
    calorie_tolerance_percent: float = Query(0.15, ge=0.0, le=0.5, description="Toleransi persentase kalori (misal 0.1 untuk 10%)")
):
    if not data_loaded_successfully or generator_instance is None:
        raise HTTPException(status_code=503, detail="Layanan tidak siap. Sumber daya gagal dimuat.")

    try:
        meal_plans = generator_instance.generate_meal_plans(
            total_calories_target=total_calories,
            max_plans=max_plans,
            calorie_tolerance_percent=calorie_tolerance_percent
        )

        if not meal_plans:
            return {"meal_plans": [], "message": "Tidak ada rencana makan yang sesuai ditemukan. Coba longgarkan toleransi atau target kalori yang berbeda."}
        
        return {"meal_plans": meal_plans}

    except Exception as e:
        print(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal saat menghasilkan rencana makan.")

@app.get("/")
async def read_root():
    return {"message": "Selamat datang di Meal Plan Generator API. Gunakan endpoint /docs untuk melihat dokumentasi API."}