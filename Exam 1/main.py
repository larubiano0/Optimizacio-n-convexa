from numpy import array, dot, zeros, sqrt
from numpy.linalg import norm
from numpy.random import seed, rand
from pandas import read_csv, concat, DataFrame
from time import perf_counter
from math import isclose
from random import sample
from matplotlib import pyplot as plt


s=10
r=3
n = 784
seed(123456) # Set seed for reproducibility

# Load train 
train = read_csv('train.csv')

# Randomly select 1000 samples of each class for training
train = concat([train[train['label'] == i].sample(1000) for i in range(s)])
labels = train['label'].values
images = train.drop('label', axis=1).values
# Regularize images with norm 3
images = array([r * image / norm(image) for image in images])

N = len(train)

# Load test
test = read_csv('test.csv')

def subgradient(X, images, labels):
    subgradient = array([zeros(n) for _ in range(s)])
    for image, label in zip(images, labels):
        active_set = []
        for m in range(s):
            val_m = 1 + dot(image, X[m] - X[label])
            if val_m > 0:
                active_set.append((m, val_m))
        maximum_in_active_set = max(active_set, key=lambda x: x[1])[1]
        # Remove elements that are less than the maximum
        active_set_new = []
        for m, val_m in active_set:
            if isclose(val_m, maximum_in_active_set):
                active_set_new.append((m, val_m))
        # If active set is empty, do nothing, otherwise, update subgradient
        if len(active_set) == 0:
            continue
        else: 
            for m, val_m in active_set:
                subgradient[m] += image 
            subgradient[label] -= image
    subgradient = 1/N * subgradient
    return subgradient

def stochastic_subgradient(X, images, labels):
    batch_size = 10
    subgradient = array([zeros(n) for _ in range(s)])
    # Select 10 random indices from 0 to 9999, sample
    batch = sample(range(N), batch_size)
    for image, label, i in zip(images, labels, range(N)):
        if i not in batch:
            continue
        active_set = []
        for m in range(s):
            val_m = 1 + dot(image, X[m] - X[label])
            if val_m > 0:
                active_set.append((m, val_m))
        maximum_in_active_set = max(active_set, key=lambda x: x[1])[1]
        # Remove elements that are less than the maximum
        active_set_new = []
        for m, val_m in active_set:
            if isclose(val_m, maximum_in_active_set):
                active_set_new.append((m, val_m))
        # If active set is empty, do nothing, otherwise, update subgradient
        if len(active_set) == 0:
            continue
        else: 
            for m, val_m in active_set:
                subgradient[m] += image 
            subgradient[label] -= image
    subgradient = 1/N * subgradient
    return subgradient

def loss_function(X, a, l):
    values = []
    for m in range(s):
        if m != l:
            values.append(max(0, 1 + dot(a, X[m]-X[l])))
    return max(values)

def objective_function(X, images, labels):
    total_sum = 0
    for image, label in zip(images, labels):
        total_sum += loss_function(X, image, label)
    return total_sum/N

def objective_function_for_Xm(Xm, images, labels):
    total_sum = 0
    for image, label in zip(images, labels):
        total_sum += loss_function(X, image, label)
    return total_sum/1000

# 5 minutes time limit
# Projected subgradient method 


alphas = [10, 1, 0.1, 0.01]

Fs_for_each = []

for alpha in alphas:
    Fs = []
    start = perf_counter()
    print(f'Alpha: {alpha}')
    k = 0
    X = array([rand(n) for _ in range(s)])
    for m in range(s):
        X[m] = r * X[m] / norm(X[m])

    while perf_counter() - start < 300: # 5 minutes
        a_k = alpha / sqrt(k+1)
        subgradient_at_X = subgradient(X, images, labels)
        for m in range(s):
            X[m] = X[m] - a_k * subgradient_at_X[m]
        # Project each X[m] to the unit ball
        for m in range(s):
            norm_X_m = norm(X[m])
            if norm_X_m <= r:
                continue
            else:
                X[m] = r * X[m] / norm_X_m
        if k % 50 == 0:
            print(f'Iteration {k} completed')
            print(f'Objective function value: {objective_function(X, images, labels)}')
        k += 1
        Fs.append(objective_function(X, images, labels))
    Fs_for_each.append(Fs)


# 5 minutes time limit
# Projected subgradient with Polyak step-size

print("\n"*3)
f_star = 0.62988 # Optimal value of the objective function from testing
Fs_Polyak = []
start = perf_counter()
k = 0
X = array([rand(n) for _ in range(s)])
for m in range(s):
    X[m] = r * X[m] / norm(X[m])

while perf_counter() - start < 300: # 5 minutes
    a_k = (objective_function(X, images, labels) - f_star) / (norm(subgradient(X, images, labels))**2)
    subgradient_at_X = subgradient(X, images, labels)

    for m in range(s):
        X[m] = X[m] - a_k * subgradient_at_X[m]
    # Project each X[m] to the unit ball
    for m in range(s):
        norm_X_m = norm(X[m])
        if norm_X_m <= r:
            continue
        else:
            X[m] = r * X[m] / norm_X_m
    if k % 50 == 0:
        print(f'Iteration {k} completed')
        print(f'Objective function value: {objective_function(X, images, labels)}')
    k += 1
    Fs_Polyak.append(objective_function(X, images, labels))

# 5 minutes time limit
# Stochastic subgradient method 

print("\n"*3)

Rs = [10, 1, 0.1, 0.01]
Fs_stochastic = []

for R in Rs:
    Fs = []
    start = perf_counter()
    print(f'R: {R}')
    k = 0
    X = array([rand(n) for _ in range(s)])
    for m in range(s):
        X[m] = r * X[m] / norm(X[m])

    while perf_counter() - start < 300: # 5 minutes
        lipschitz_constants = []
        subgradient_at_X = stochastic_subgradient(X, images, labels)
        for m in range(s):
            a_k =  R / (sqrt(k+1) * norm(subgradient_at_X[m]))
            X[m] = X[m] - a_k * subgradient_at_X[m]
        # Project each X[m] to the unit ball
        for m in range(s):
            norm_X_m = norm(X[m])
            if norm_X_m <= r:
                continue
            else:
                X[m] = r * X[m] / norm_X_m
        if k % 50 == 0:
            print(f'Iteration {k} completed')
            print(f'Objective function value: {objective_function(X, images, labels)}')
        k += 1
        Fs.append(objective_function(X, images, labels))
    Fs_stochastic.append(Fs)

# Save the results in csv
df = DataFrame(Fs_for_each)
df.to_csv('Fs_for_each.csv')

df = DataFrame(Fs_Polyak)
df.to_csv('Fs_Polyak.csv')

df = DataFrame(Fs_stochastic)
df.to_csv('Fs_stochastic.csv')


## Testing

X = array([zeros(n) for _ in range(s)])
for _ in range(1000):
    for m in range(s):
        X[m] += images[1000*m + _]
X = X / 1000
print(f'Objective function value: {objective_function(X, images, labels)}')

