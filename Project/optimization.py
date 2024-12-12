import pandas as pd
from scipy.optimize import linprog

# Load data
file_path = "nutrition_filtered_grams_prices.csv"
data = pd.read_csv(file_path)
if data.isnull().any().any():
    #Change missing values to 0
    data.fillna(0, inplace=True)
names = data['name'].values
calories = data['calories'].values
protein = data['protein'].values
total_fat = data['total_fat'].values
saturated_fat = data['saturated_fat'].values
carbs = data['carbohydrate'].values
fiber = data['fiber'].values
sugar = data['sugars'].values
sodium = data['sodium'].values
prices = data['price'].values

# Nutrient constraints
c_min, c_max = 2200, 2400
p_min, p_max = 140, 150
f_min, f_max = 60, 70
sf_min, sf_max = 0, 25
ca_min, ca_max = 240, 300
fi_min, fi_max = 25, 35
su_min, su_max = 0, 50
so_min, so_max = 1, 2.3

# Number of products
n = len(names)

# Define the linear programming problem
# Objective function: minimize total price (prices^T * x)

# Inequality constraints (Ax <= b)
A = [
    calories,           
    -calories,          
    protein,            
    -protein,          
    total_fat,          
    -total_fat,         
    saturated_fat,      
    -saturated_fat,     
    carbs,              
    -carbs,             
    fiber,              
    -fiber,             
    sugar,              
    -sugar,             
    sodium,             
    -sodium             
]
b = [
    c_max, -c_min,      
    p_max, -p_min,      
    f_max, -f_min,     
    sf_max, -sf_min,    
    ca_max, -ca_min,    
    fi_max, -fi_min,    
    su_max, -su_min,    
    so_max, -so_min     
]

# Bounds for variables 
x_bounds = [(0, 1) for _ in range(n)]  # Non-negative quantities, at most 100 grams of each product

# Solve the linear programming problem
result = linprog(prices, A_ub=A, b_ub=b, bounds=x_bounds, method='simplex')

# Check the result
if result.success:
    print("Optimal solution found")
    solution = result.x
    total_cost = result.fun
    selected_items = {names[i]: (solution[i], prices[i]) for i in range(n) if solution[i] > 0}
    print("Selected items:")
    for name, (quantity, price) in selected_items.items():
        print(f"{name}: {quantity*100:.2f} grams, price: {price*quantity:.2f}")
    print(f"Total cost: {total_cost:.2f}")
    print("Nutritional values of the selected items:")
    total_calories = sum(calories[i] * solution[i] for i in range(len(solution)))
    total_protein = sum(protein[i] * solution[i] for i in range(len(solution)))
    total_fat = sum(total_fat[i] * solution[i] for i in range(len(solution)))
    total_saturated_fat = sum(saturated_fat[i] * solution[i] for i in range(len(solution)))
    total_carbs = sum(carbs[i] * solution[i] for i in range(len(solution)))
    total_fiber = sum(fiber[i] * solution[i] for i in range(len(solution)))
    total_sugar = sum(sugar[i] * solution[i] for i in range(len(solution)))
    total_sodium = sum(sodium[i] * solution[i] for i in range(len(solution)))
    print(f"Total calories: {total_calories:.2f}")
    print(f"Total protein: {total_protein:.2f}")
    print(f"Total fat: {total_fat:.2f}")
    print(f"Total saturated fat: {total_saturated_fat:.2f}")
    print(f"Total carbs: {total_carbs:.2f}")
    print(f"Total fiber: {total_fiber:.2f}")
    print(f"Total sugar: {total_sugar:.2f}")
    print(f"Total sodium: {total_sodium:.2f}")
else:
    print("No feasible solution found.")