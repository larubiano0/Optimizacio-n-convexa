import pandas as pd

# Load the data
foods = pd.read_csv('nutrition_filtered.csv') # Each food's serving size is 100 grams

# Keep only columns calories, protein, total_fat, saturated_fat, carbohydrate,fiber,sugars and sodium

foods = foods[['name', 'calories', 'protein', 'total_fat', 'saturated_fat', 'carbohydrate', 'fiber', 'sugars', 'sodium']]

# Set all the values to grams 

def convert_to_grams(value):
    if type(value) != float:
        value = value.replace(' ', '')
    if pd.isnull(value):
        return value
    elif value.endswith('mg'):
        return float(value[:-2]) / 1000
    elif value.endswith('g'):
        return float(value[:-1])
    else:
        return 0
    
foods['protein'] = foods['protein'].apply(convert_to_grams)
foods['total_fat'] = foods['total_fat'].apply(convert_to_grams)
foods['saturated_fat'] = foods['saturated_fat'].apply(convert_to_grams)
foods['carbohydrate'] = foods['carbohydrate'].apply(convert_to_grams)
foods['fiber'] = foods['fiber'].apply(convert_to_grams)
foods['sugars'] = foods['sugars'].apply(convert_to_grams)
foods['sodium'] = foods['sodium'].apply(convert_to_grams)

# Save the data

foods.to_csv('nutrition_filtered_grams.csv', index=False)
