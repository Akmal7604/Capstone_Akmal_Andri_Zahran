import pandas as pd
import joblib
from typing import Union
from fastapi import FastAPI, Query
import numpy as np

# Load model yang sudah disimpan
generator = joblib.load('meal_plan_generator.joblib')
kmeans_model = joblib.load('kmeans_model.joblib')

# Buat FastAPI app
app = FastAPI(
    title="Meal Plan Generator",
    description="Generate personalized meal plans based on total calories",
    version="1.0.0"
)

@app.get("/", summary="Root endpoint")
def read_root():
    return {"message": "Welcome to Meal Plan Generator API"}

@app.get("/generate_meal_plan", 
         summary="Generate Meal Plan",
         description="Generate meal plans based on total calories")
def generate_meal_plan(
    total_calories: int = Query(default=1000, ge=100, le=3000, 
                                description="Total calories for meal plan (100-3000)")
):
    """
    Generate meal plan with RecipeId, Name, and total calories
    
    - total_calories: Target total calories for the meal plan
    - Returns list of meal plans with RecipeId and Name
    """
    try:
        # Generate meal plan
        meal_plans = generator.generate_meal_plans(total_calories)
        
        # Format output dengan RecipeId dan Name
        output_meal_plans = []
        for plan in meal_plans:
            output_plan = {
                'Breakfast': {
                    'RecipeId': generator.df.loc[generator.df['Name'] == plan['Breakfast']['Name'], 'RecipeId'].values[0],
                    'Name': plan['Breakfast']['Name'],
                    'Calories': plan['Breakfast']['Calories']
                },
                'Lunch': {
                    'RecipeId': generator.df.loc[generator.df['Name'] == plan['Lunch']['Name'], 'RecipeId'].values[0],
                    'Name': plan['Lunch']['Name'],
                    'Calories': plan['Lunch']['Calories']
                },
                'Dinner': {
                    'RecipeId': generator.df.loc[generator.df['Name'] == plan['Dinner']['Name'], 'RecipeId'].values[0],
                    'Name': plan['Dinner']['Name'],
                    'Calories': plan['Dinner']['Calories']
                },
                'Total Calories': plan['Total Calories']
            }
            output_meal_plans.append(output_plan)
        
        return {
            "status": "success",
            "meal_plans": output_meal_plans
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Untuk menjalankan:
# uvicorn main:app --reload
