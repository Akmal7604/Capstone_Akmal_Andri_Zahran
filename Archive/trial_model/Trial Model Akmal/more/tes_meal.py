import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
# from ast import literal_eval # Not strictly needed with the current parse_keywords
import joblib # For saving models

# Load the dataframe
# Make sure the path to your CSV is correct and that it contains a 'RecipeId' column
try:
    # Ensure 'RecipeId' is one of the columns in your CSV.
    # If it's named differently, please adjust the code in generate_meal_plans method.
    df = pd.read_csv("D:\\UB\\Dicoding\\GitHub\\Capstone_Akmal_Andri_Zahran\\recipes_new.csv")
    # Minimal check if critical columns exist
    required_columns = ['RecipeId', 'Name', 'Keywords', 'Calories', 'ProteinContent', 'CarbohydrateContent']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing}. Please ensure your CSV has these columns.")
    
    # Convert relevant columns to numeric, coercing errors
    numeric_cols = ['Calories', 'ProteinContent', 'CarbohydrateContent']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in essential numeric columns after coercion
    df.dropna(subset=numeric_cols, inplace=True)
    
    # Fill NaN in 'Keywords' with an empty string if any, to prevent errors in parse_keywords
    df['Keywords'] = df['Keywords'].fillna('')

except FileNotFoundError:
    print("Error: The CSV file was not found. Please check the path.")
    exit()
except ValueError as ve:
    print(ve)
    exit()


class MealPlanGenerator:
    def __init__(self, df):
        self.df = df.copy() # Use a copy to avoid SettingWithCopyWarning later
        self.scaler = StandardScaler() # Initialize scaler here
        self.kmeans = None # Initialize kmeans here, will be fitted in cluster_recipes
        self.keyword_columns = []
        self.preprocess_data()
        self.cluster_recipes()

    def preprocess_data(self):
        # Preprocessing untuk kategorisasi keywords
        def parse_keywords(keywords):
            try:
                # Konversi string ke list
                if isinstance(keywords, str):
                    # Handles empty strings or strings of spaces as well
                    return [k.strip() for k in keywords.split(',') if k.strip()]
                elif isinstance(keywords, list): # if it's already a list
                    return [str(k).strip() for k in keywords if str(k).strip()]
                return [] # Default to empty list for other types or parsing issues
            except Exception: # Catch any parsing error
                return []

        # Kategorisasi waktu makan
        meal_time_keywords = {
            'Breakfast': ['Breakfast', 'Brunch', 'Oatmeal', 'Eggs'],
            'Lunch': ['Lunch/Snacks', 'Salad', 'Sandwich', 'Lunch'], # Added 'Lunch'
            'Dinner': ['Dinner', 'Meat', 'Chicken', 'Fish', 'Steak', 'Stir Fry']
        }

        # Parse keywords
        self.df['ParsedKeywords'] = self.df['Keywords'].apply(parse_keywords)

        # Kategorisasi meal type
        def categorize_meal_type(parsed_keywords_list):
            if not isinstance(parsed_keywords_list, list): # Ensure it's a list
                return 'Other'
            for meal, meal_keywords in meal_time_keywords.items():
                if any(kw in parsed_keywords_list for kw in meal_keywords):
                    return meal
            return 'Other'

        self.df['MealType'] = self.df['ParsedKeywords'].apply(categorize_meal_type)

        # Multi-label encoding untuk keywords (optional for current output, but kept from original)
        # Ensure all elements in ParsedKeywords are lists of strings
        valid_keywords = self.df['ParsedKeywords'].apply(lambda x: [str(item) for item in x] if isinstance(x, list) else [])
        
        mlb = MultiLabelBinarizer()
        # Fit transform on valid keywords. It's okay if some are empty lists.
        keyword_encoded = mlb.fit_transform(valid_keywords)
        self.keyword_columns = mlb.classes_
        
        # For debugging: Check meal type distribution
        # print("Meal type distribution after preprocessing:")
        # print(self.df['MealType'].value_counts())

    def cluster_recipes(self):
        # Fitur untuk clustering
        features = self.df[['Calories', 'ProteinContent', 'CarbohydrateContent']]
        
        # Normalisasi fitur (scaler initialized in __init__)
        features_scaled = self.scaler.fit_transform(features)

        # Clustering menggunakan KMeans
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto') # Explicitly set n_init
        self.df['Cluster'] = self.kmeans.fit_predict(features_scaled)

    def generate_meal_plans(self, total_calories_target, max_plans=4, calorie_tolerance_percent=0.10):
        meal_plans = []
        
        # Filter kategori makanan
        breakfast_df = self.df[self.df['MealType'] == 'Breakfast']
        lunch_df = self.df[self.df['MealType'] == 'Lunch']
        dinner_df = self.df[self.df['MealType'] == 'Dinner']

        # Check if any category is empty
        if breakfast_df.empty or lunch_df.empty or dinner_df.empty:
            print("Warning: Not enough variety in meal types (Breakfast, Lunch, or Dinner) to generate plans.")
            if breakfast_df.empty: print("No breakfast items found.")
            if lunch_df.empty: print("No lunch items found.")
            if dinner_df.empty: print("No dinner items found.")
            return meal_plans # Return empty list

        attempts = 0
        max_attempts = max_plans * 50 # Limit attempts to avoid infinite loops

        while len(meal_plans) < max_plans and attempts < max_attempts:
            attempts += 1
            try:
                # Sampling dengan mempertimbangkan cluster (original logic, can be enhanced)
                breakfast = breakfast_df.sample(1).iloc[0]
                lunch = lunch_df.sample(1).iloc[0]
                dinner = dinner_df.sample(1).iloc[0]
            except ValueError:
                # This can happen if, after filtering, a df becomes empty unexpectedly
                # or if sample(1) is called on an empty df (though checked above).
                print("Warning: Could not sample a meal, a category might be unexpectedly empty.")
                continue # Skip this iteration

            total_plan_calories = (
                breakfast['Calories'] +
                lunch['Calories'] +
                dinner['Calories']
            )
            
            # Validasi total kalori (e.g., within a certain percentage of the target)
            # Allow for some flexibility, e.g., 10% below and 5% above
            lower_bound = total_calories_target * (1 - calorie_tolerance_percent)
            # upper_bound = total_calories_target * (1 + (calorie_tolerance_percent / 2)) # Stricter upper bound

            if lower_bound <= total_plan_calories <= total_calories_target : # Only accept plans AT or BELOW target
                meal_plan = {
                    'Breakfast': {
                        'RecipeId': breakfast['RecipeId'], # Added RecipeId
                        'Name': breakfast['Name'],
                        'Calories': breakfast['Calories']
                        # 'Keywords': breakfast['ParsedKeywords'] # Removed as per new output requirement
                    },
                    'Lunch': {
                        'RecipeId': lunch['RecipeId'], # Added RecipeId
                        'Name': lunch['Name'],
                        'Calories': lunch['Calories']
                        # 'Keywords': lunch['ParsedKeywords'] # Removed
                    },
                    'Dinner': {
                        'RecipeId': dinner['RecipeId'], # Added RecipeId
                        'Name': dinner['Name'],
                        'Calories': dinner['Calories']
                        # 'Keywords': dinner['ParsedKeywords'] # Removed
                    },
                    'TotalCalories': total_plan_calories # Consistent naming
                }
                # Avoid duplicate plans (based on the combination of recipe names for simplicity)
                plan_signature = (breakfast['RecipeId'], lunch['RecipeId'], dinner['RecipeId'])
                if not any(p['Breakfast']['RecipeId'] == plan_signature[0] and \
                           p['Lunch']['RecipeId'] == plan_signature[1] and \
                           p['Dinner']['RecipeId'] == plan_signature[2] for p in meal_plans):
                    meal_plans.append(meal_plan)
            
        if attempts >= max_attempts and len(meal_plans) < max_plans:
            print(f"Warning: Could only generate {len(meal_plans)} plans after {max_attempts} attempts. Consider adjusting calorie targets or dataset.")

        return meal_plans

    def display_meal_plans(self, meal_plans):
        if not meal_plans:
            print("No meal plans generated.")
            return
        for i, plan in enumerate(meal_plans, 1):
            print(f"--- Meal Plan {i} ---")
            for meal_type, meal_info in plan.items():
                if meal_type != 'TotalCalories':
                    print(f"  {meal_type}:")
                    print(f"    RecipeId: {meal_info['RecipeId']}")
                    print(f"    Name: {meal_info['Name']}")
                    print(f"    Calories: {meal_info['Calories']}")
            print(f"  Total Plan Calories: {plan['TotalCalories']}\n")

    def save_models(self, kmeans_path="kmeans_model.joblib", scaler_path="scaler.joblib"):
        """Saves the K-Means model and the StandardScaler using joblib."""
        if self.kmeans is not None:
            joblib.dump(self.kmeans, kmeans_path)
            print(f"KMeans model saved to {kmeans_path}")
        else:
            print("KMeans model has not been trained yet. Cannot save.")
        
        if self.scaler is not None: # Scaler is always fitted if preprocess_data runs
            joblib.dump(self.scaler, scaler_path)
            print(f"StandardScaler saved to {scaler_path}")
        else:
            # This case should ideally not be reached if __init__ logic is correct
            print("Scaler has not been initialized or fitted. Cannot save.")

# --- Example Usage ---
if __name__ == "__main__":
    if 'df' in globals() and not df.empty: # Check if df was loaded successfully
        # Inisialisasi generator
        generator = MealPlanGenerator(df)

        # --- This part simulates input from a front-end ---
        try:
            user_total_calories = float(input("Enter your desired total daily calories (e.g., 1500, 2000): "))
            if user_total_calories <= 0:
                print("Calories must be a positive number.")
                exit()
        except ValueError:
            print("Invalid input. Please enter a number for calories.")
            exit()
        # --- End of simulated front-end input ---

        print(f"\nGenerating meal plans for approximately {user_total_calories} calories...")
        # Generate meal plan with user-defined total calories
        # You can also adjust max_plans and calorie_tolerance_percent if needed
        meal_plans_output = generator.generate_meal_plans(user_total_calories, max_plans=3, calorie_tolerance_percent=0.20)

        # Tampilkan meal plan (new format)
        generator.display_meal_plans(meal_plans_output)

        # Simpan model untuk deployment
        # These files (kmeans_model.joblib, scaler.joblib) will be created in your current working directory.
        generator.save_models()

        print("\n--- Raw Output (for API integration) ---")
        # This is how you might want to structure the JSON output for an API
        # The meal_plans_output is already in a suitable list-of-dictionaries format
        import json
        print(json.dumps(meal_plans_output, indent=2))

    else:
        print("DataFrame 'df' is not loaded or is empty. Cannot proceed.")