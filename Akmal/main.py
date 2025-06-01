import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
from typing import List, Union, Optional # << TAMBAHKAN INI

# --- Bagian 1: Definisi Kelas MealPlanGenerator ---
# (Ini adalah kelas yang sudah Anda buat, dengan sedikit modifikasi untuk memuat model)

class MealPlanGenerator:
    def __init__(self, df, preloaded_scaler=None, preloaded_kmeans=None):
        self.df = df.copy() # Gunakan salinan untuk menghindari SettingWithCopyWarning
        self.keyword_columns = [] # Akan diisi oleh preprocess_data

        # Preprocess data (seperti kategorisasi MealType)
        self.preprocess_data()

        if preloaded_scaler and preloaded_kmeans:
            self.scaler = preloaded_scaler
            self.kmeans = preloaded_kmeans
            # Tetapkan cluster menggunakan model yang sudah dimuat
            self._assign_clusters_from_loaded_model()
        else:
            # Perilaku asli: latih scaler dan kmeans baru
            self.scaler = StandardScaler()
            # Inisialisasi kmeans di sini, akan di-fit di _cluster_recipes_and_fit
            self.kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
            self._cluster_recipes_and_fit()

    def preprocess_data(self):
        def parse_keywords(keywords):
            try:
                if isinstance(keywords, str):
                    return [k.strip() for k in keywords.split(',') if k.strip()]
                elif isinstance(keywords, list):
                    return [str(k).strip() for k in keywords if str(k).strip()]
                return []
            except Exception:
                return []

        meal_time_keywords = {
            'Breakfast': ['Breakfast', 'Brunch', 'Oatmeal', 'Eggs'],
            'Lunch': ['Lunch/Snacks', 'Salad', 'Sandwich', 'Lunch'],
            'Dinner': ['Dinner', 'Meat', 'Chicken', 'Fish', 'Steak', 'Stir Fry']
        }

        self.df['ParsedKeywords'] = self.df['Keywords'].apply(parse_keywords)

        def categorize_meal_type(parsed_keywords_list):
            if not isinstance(parsed_keywords_list, list):
                return 'Other'
            for meal, meal_keywords in meal_time_keywords.items():
                if any(kw in parsed_keywords_list for kw in meal_keywords):
                    return meal
            return 'Other'

        self.df['MealType'] = self.df['ParsedKeywords'].apply(categorize_meal_type)
        
        # Pastikan semua ParsedKeywords adalah list of strings untuk MLB
        valid_keywords = self.df['ParsedKeywords'].apply(lambda x: [str(item) for item in x] if isinstance(x, list) else [])
        mlb = MultiLabelBinarizer()
        # Tidak perlu menyimpan keyword_encoded secara global jika tidak dipakai fitur clustering lain
        mlb.fit_transform(valid_keywords) 
        self.keyword_columns = mlb.classes_

    def _cluster_recipes_and_fit(self):
        # Metode ini untuk melatih model (digunakan jika model tidak di-preload)
        features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        if features.isnull().values.any():
            # Lakukan penanganan NaN sebelum scaling jika ada (misal, imputasi atau drop)
            # Contoh: self.df.dropna(subset=['Calories', 'ProteinContent', 'CarbohydrateContent'], inplace=True)
            #         features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
            #         jika dilakukan di sini, pastikan self.df konsisten
            print("Warning: NaN values found in features for clustering. Ensure data is clean.")
            # Untuk production, sebaiknya raise error atau handle lebih baik
            self.df.dropna(subset=['Calories', 'ProteinContent', 'CarbohydrateContent'], inplace=True)
            features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]


        features_scaled = self.scaler.fit_transform(features)
        self.df['Cluster'] = self.kmeans.fit_predict(features_scaled)

    def _assign_clusters_from_loaded_model(self):
        # Metode ini untuk menetapkan cluster menggunakan model yang sudah dimuat
        features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        if features.isnull().values.any():
            print("Warning: NaN values found in features for clustering during assignment. Ensure data is clean.")
            self.df.dropna(subset=['Calories', 'ProteinContent', 'CarbohydrateContent'], inplace=True) # sinkronkan df
            features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]

        features_scaled = self.scaler.transform(features) # Penting: gunakan transform, bukan fit_transform
        self.df['Cluster'] = self.kmeans.predict(features_scaled)


    def generate_meal_plans(self, total_calories_target, max_plans=4, calorie_tolerance_percent=0.10):
        meal_plans = []
        
        breakfast_df = self.df[self.df['MealType'] == 'Breakfast']
        lunch_df = self.df[self.df['MealType'] == 'Lunch']
        dinner_df = self.df[self.df['MealType'] == 'Dinner']

        if breakfast_df.empty or lunch_df.empty or dinner_df.empty:
            # Tidak cukup variasi untuk membuat rencana, kembalikan list kosong
            return meal_plans 

        attempts = 0
        # Tingkatkan max_attempts jika diperlukan, terutama jika target kalori sangat spesifik
        max_attempts = max(max_plans * 100, 1000) 

        generated_plan_signatures = set()

        while len(meal_plans) < max_plans and attempts < max_attempts:
            attempts += 1
            try:
                breakfast = breakfast_df.sample(1).iloc[0]
                lunch = lunch_df.sample(1).iloc[0]
                dinner = dinner_df.sample(1).iloc[0]
            except ValueError: # Terjadi jika salah satu df (breakfast/lunch/dinner) kosong
                continue 

            total_plan_calories = breakfast['Calories'] + lunch['Calories'] + dinner['Calories']
            
            lower_bound = total_calories_target * (1 - calorie_tolerance_percent)
            # upper_bound = total_calories_target * (1 + (calorie_tolerance_percent / 2)) # Batas atas yang lebih ketat

            # Hanya terima rencana yang kalorinya DI BAWAH atau SAMA DENGAN target, tapi dalam toleransi
            if lower_bound <= total_plan_calories <= total_calories_target:
                # Buat signature unik untuk rencana ini untuk menghindari duplikat
                plan_signature = (breakfast['RecipeId'], lunch['RecipeId'], dinner['RecipeId'])
                if plan_signature in generated_plan_signatures:
                    continue # Lewati jika rencana ini sudah ada

                meal_plan = {
                    'Breakfast': {'RecipeId': breakfast['RecipeId'], 'Name': breakfast['Name'], 'Calories': breakfast['Calories']},
                    'Lunch': {'RecipeId': lunch['RecipeId'], 'Name': lunch['Name'], 'Calories': lunch['Calories']},
                    'Dinner': {'RecipeId': dinner['RecipeId'], 'Name': dinner['Name'], 'Calories': dinner['Calories']},
                    'TotalCalories': total_plan_calories
                }
                meal_plans.append(meal_plan)
                generated_plan_signatures.add(plan_signature)
        
        return meal_plans

    # Metode save_models dan display_meal_plans tidak diperlukan untuk FastAPI runtime,
    # tapi bisa disimpan jika kelas ini juga digunakan untuk training.
    def save_models(self, kmeans_path="kmeans_model.joblib", scaler_path="scaler.joblib"):
        if self.kmeans is not None: joblib.dump(self.kmeans, kmeans_path)
        if self.scaler is not None: joblib.dump(self.scaler, scaler_path)


# --- Bagian 2: Aplikasi FastAPI ---
app = FastAPI(
    title="Meal Plan Generator API",
    description="API untuk menghasilkan rencana makan berdasarkan target kalori.",
    version="1.0.0"
)

# Variabel global untuk menyimpan instance generator dan data yang dimuat
generator_instance: Optional[MealPlanGenerator] = None # << Perubahan kecil untuk kejelasan
data_loaded_successfully = False

# Path ke file (sesuaikan jika perlu)
# Sebaiknya gunakan path relatif atau variabel lingkungan untuk deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Direktori tempat main.py berada
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(BASE_DIR, "kmeans_model.joblib")
# Ganti dengan path CSV Anda yang benar jika tidak di BASE_DIR
# Pastikan path ini benar di lingkungan deployment Anda
# CSV_PATH = os.path.join(BASE_DIR, "recipes_new.csv")
CSV_PATH = "D:\\UB\\Dicoding\\GitHub\\Capstone_Akmal_Andri_Zahran\\recipes_new.csv"


@app.on_event("startup")
async def load_resources():
    global generator_instance, data_loaded_successfully
    try:
        print(f"Mencoba memuat scaler dari: {SCALER_PATH}")
        loaded_scaler = joblib.load(SCALER_PATH)
        print(f"Mencoba memuat kmeans dari: {KMEANS_PATH}")
        loaded_kmeans = joblib.load(KMEANS_PATH)
        print(f"Mencoba memuat CSV dari: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)

        # Pembersihan data minimal (konsisten dengan skrip training)
        required_cols = ['RecipeId', 'Name', 'Keywords', 'Calories', 'ProteinContent', 'CarbohydrateContent']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise FileNotFoundError(f"Kolom yang dibutuhkan tidak ada di CSV: {missing}")

        numeric_cols = ['Calories', 'ProteinContent', 'CarbohydrateContent']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop baris dengan NaN di kolom numerik esensial SETELAH preprocessing dasar di generator
        # Penanganan NaN lebih baik dilakukan di dalam kelas atau sebelum memanggil kelas
        df['Keywords'] = df['Keywords'].fillna('')
        # Penting: drop NaN *sebelum* mengirim ke generator jika scaler/kmeans tidak bisa handle
        df.dropna(subset=numeric_cols, inplace=True)
        if df.empty:
            raise ValueError("DataFrame kosong setelah membersihkan NaN dari kolom numerik esensial.")


        # Inisialisasi generator dengan model yang sudah dimuat
        generator_instance = MealPlanGenerator(df, preloaded_scaler=loaded_scaler, preloaded_kmeans=loaded_kmeans)
        data_loaded_successfully = True
        print("Sumber daya berhasil dimuat dan MealPlanGenerator diinisialisasi dengan model yang sudah ada.")

    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan saat startup: {e}")
        data_loaded_successfully = False
    except ValueError as e:
        print(f"Error: Nilai tidak valid saat startup: {e}")
        data_loaded_successfully = False
    except Exception as e:
        print(f"Error tidak terduga saat startup: {e}")
        data_loaded_successfully = False


# Definisikan model respons untuk dokumentasi API yang lebih baik (opsional tapi bagus)
class MealDetail(BaseModel):
    RecipeId: int # Atau str, tergantung tipe data Anda
    Name: str
    Calories: float

class MealPlanResponse(BaseModel):
    Breakfast: MealDetail
    Lunch: MealDetail
    Dinner: MealDetail
    TotalCalories: float

class MealPlansListResponse(BaseModel):
    meal_plans: List[MealPlanResponse]     # << UBAH INI
    message: Optional[str] = None          # << UBAH INI


@app.get(
    "/generate-meal-plan/",
    response_model=MealPlansListResponse, # Menggunakan Pydantic model untuk respons
    summary="Generate Meal Plans",
    description="Menghasilkan beberapa rencana makan berdasarkan target kalori harian."
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
            return {"meal_plans": [], "message": "Tidak ada rencana makan yang sesuai ditemukan untuk target kalori yang diberikan."}
        
        return {"meal_plans": meal_plans}

    except ValueError as ve: # Misalnya jika input tidak valid yang tidak tertangkap Query
        raise HTTPException(status_code=400, detail=f"Input tidak valid: {ve}")
    except Exception as e:
        # Sebaiknya log error 'e' di sini untuk debugging
        print(f"Internal server error: {e}") # Untuk konsol server
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal saat menghasilkan rencana makan.")

@app.get("/")
async def read_root():
    return {"message": "Selamat datang di Meal Plan Generator API. Gunakan endpoint /docs untuk melihat dokumentasi API."}

# Untuk menjalankan (jika file ini disimpan sebagai main.py):
# fastapi dev main.py