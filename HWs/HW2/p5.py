import matplotlib.pyplot as plt
import os
from time import perf_counter
from numpy import float16, dot, ones, zeros, log
from numpy.linalg import norm, solve
import pandas as pd
from progress.bar import Bar

n = 5000
m = 2000
epsilon = float16(1e-6)
max_iterations = 6 # Maximum number of iterations
max_sub_iterations = 3 # Maximum number of sub-iterations
# C is the vector of ones
ONE = float16(1)
TWO = float16(2)

dots = {}
def optimized_dot(x,y):
    stringxy = str(x) + str(y)
    if stringxy in dots:
        return dots[stringxy]
    result = dot(x,y)
    dots[stringxy] = result
    return result

fs = {}
def f(x, A):
    stringx = str(x)
    if stringx in fs:
        return fs[stringx]

    result = sum(x)
    for j in range(m):
        result -= log(ONE - optimized_dot(A[j],x))
    for i in range(n):
        result -= log(ONE - x[i]*x[i])

    fs[stringx] = result
    return result

gradients = {}
def gradient(x, A):
    stringx = str(x)
    if stringx in gradients:
        return gradients[stringx]

    result = ones(n, dtype=float16)
    with Bar('Gradient', fill='#', suffix='%(percent).1f%% - %(eta)ds', max=n) as bar:
        for i in range(n):
            for j in range(m):
                result[i] += A[j][i] / (ONE - optimized_dot(A[j],x))
            result[i] += TWO * x[i] / (ONE - x[i]*x[i])
            bar.next()

    gradients[stringx] = result
    return result

hessians = {}
def hessian(x, A):
    # We assume its symmetric
    stringx = str(x)
    if stringx in hessians:
        return hessians[stringx]

    result = zeros((n,n), dtype=float16)
    with Bar('Hessian', fill='#', suffix='%(percent).1f%% - %(eta)ds', max=n*(n+1)/2) as bar:
        for i in range(n):
            for l in range(i,n):
                for j in range(m):
                    result[i][l] += A[j][i] * A[j][l] / ((ONE - optimized_dot(A[j],x)) * (ONE - optimized_dot(A[j],x)))
                if i == l:
                    result[i][l] += TWO* (ONE + x[i]*x[i]) / ((ONE - x[i]*x[i]) * (ONE - x[i]*x[i]))

                result[l][i] = result[i][l]
                bar.next()

    hessians[stringx] = result
    return result


def graph(x, y, title, x_label, y_label, filename):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.ylabel(y_label)
    filename = "images" + os.sep + filename + ".png"
    fig.savefig(filename, dpi=fig.dpi)
    plt.clf()


A = []
df = pd.read_csv('A.txt', header=None)
data = df.values
for j in range(m):
    # Save data[j] as numpy array of float16
    A.append(data[j].astype(float16))
print("A loaded")

time0 = perf_counter() 

## Gradient method with constant step size alpha

alpha = float16(0.2) # Small constant alpha
X = [zeros(n, dtype=float16)]
Fx = [f(X[-1],A)]
P = [- gradient(X[-1], A)]
gradient_norms = [norm(P[-1])]

iterations = 0
while (gradient_norms[-1] > epsilon) and (iterations < max_iterations):
    print("Iteration", iterations)
    print("Norm of gradient:", gradient_norms[-1])
    print("f(x_k):", Fx[-1])
    print("x_k:", X[-1]) 
    iterations += 1
    X.append(X[-1] + alpha * P[-1]) # X_k+1 = X_k + alpha * P_k
    Fx.append(f(X[-1], A))
    P.append(- gradient(X[-1], A))
    gradient_norms.append(norm(P[-1]))

    

# End timer
time1 = perf_counter()

print("Gradient method with constant step size alpha =", alpha, "completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time1 - time0)/3600, "hours")

graph(range(iterations+1), Fx, "f(x) vs iterations", "Iteration", "f(x)", "5ai")
graph(range(iterations+1), gradient_norms, "Norm of gradient vs iterations", "Iteration", "Norm of gradient", "5aii")



## Gradient method with backtracking

alpha = float16(0.25) # Initial alpha
rho = float16(0.5)
c = float16(0.5)
X = [zeros(n, dtype=float16)]
Fx = [f(X[-1],A)]
ALPHAS = [alpha]
P = [- gradient(X[-1], A)]
gradient_norms = [norm(P[-1])]
iterations = 0

while (gradient_norms[-1] > epsilon) and (iterations < max_iterations):
    print("Iteration", iterations)
    print("Norm of gradient:", gradient_norms[-1])
    print("f(x_k):", Fx[-1])
    print("x_k:", X[-1]) 
    print("Alpha:", ALPHAS[-1])

    iterations += 1
    sub_iterations = 0

    while f(X[-1] + ALPHAS[-1] * P[-1], A) > f(X[-1], A) + c * ALPHAS[-1] * optimized_dot(gradient(X[-1], A), P[-1]) and sub_iterations < max_sub_iterations:
        sub_iterations += 1
        ALPHAS[-1] = rho * ALPHAS[-1]
    
    X.append(X[-1] + ALPHAS[-1] * P[-1]) # X_k+1 = X_k + alpha * P_k
    Fx.append(f(X[-1], A))
    P.append(- gradient(X[-1], A))
    gradient_norms.append(norm(P[-1]))
    ALPHAS.append(alpha)

# End timer
time2 = perf_counter()

print("Gradient method with backtraking completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time2 - time1)/3600, "hours")

graph(range(iterations+1), Fx, "f(x) vs iterations", "Iteration", "f(x)", "5bi")
graph(range(iterations+1), gradient_norms, "Norm of gradient vs iterations", "Iteration", "Norm of gradient", "5bii")
graph(range(iterations+1), ALPHAS, "Alpha vs iterations", "Iteration", "Alpha", "5biii")


## Gradient method with backtracking and Wolfe conditions

alpha = float16(0.25) # Initial alpha
c_1 = float16(10e-4)
c_2 = float16(0.9)
X = [zeros(n, dtype=float16)]
Fx = [f(X[-1],A)]
ALPHAS = [alpha]
P = [- gradient(X[-1], A)]
gradient_norms = [norm(P[-1])]
iterations = 0

while (gradient_norms[-1] > epsilon) and (iterations < max_iterations):
    print("Iteration", iterations)
    print("Norm of gradient:", gradient_norms[-1])
    print("f(x_k):", Fx[-1])
    print("x_k:", X[-1]) 
    print("Alpha:", ALPHAS[-1])

    iterations += 1
    sub_iterations = 0

    while (f(X[-1] + ALPHAS[-1] * P[-1], A) > f(X[-1], A) + c_1 * ALPHAS[-1] * optimized_dot(gradient(X[-1], A), P[-1]) or (- optimized_dot(gradient(X[-1] + ALPHAS[-1] * P[-1], A), P[-1]) > - c_2 * optimized_dot(gradient(X[-1], A), P[-1]))) and sub_iterations < max_sub_iterations:
        sub_iterations += 1
        ALPHAS[-1] = rho * ALPHAS[-1]
    
    X.append(X[-1] + ALPHAS[-1] * P[-1]) # X_k+1 = X_k + alpha * P_k
    Fx.append(f(X[-1], A))
    P.append(- gradient(X[-1], A))
    gradient_norms.append(norm(P[-1]))
    ALPHAS.append(alpha)

# End timer
time3 = perf_counter()

print("Gradient method with backtraking and Wolfe conditions completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time3 - time2)/3600, "hours")

graph(range(iterations+1), Fx, "f(x) vs iterations", "Iteration", "f(x)", "5ci")
graph(range(iterations+1), gradient_norms, "Norm of gradient vs iterations", "Iteration", "Norm of gradient", "5cii")
graph(range(iterations+1), ALPHAS, "Alpha vs iterations", "Iteration", "Alpha", "5ciii")

## Newton method with constant step size alpha

alpha = float16(0.2) # Small constant alpha
X = [zeros(n, dtype=float16)]
Fx = [f(X[-1],A)]
P = [solve(hessian(X[-1], A), -gradient(X[-1], A))]
iterations = 0

while iterations < max_iterations:
    print("Iteration", iterations)
    print("f(x_k):", Fx[-1])
    print("x_k:", X[-1]) 
    iterations += 1
    X.append(X[-1] + alpha * P[-1]) # X_k+1 = X_k + alpha * P_k
    Fx.append(f(X[-1], A))
    P.append(solve(hessian(X[-1], A), -gradient(X[-1], A)))

# End timer
time4 = perf_counter()

print("Newton method with constant step size alpha =", alpha, "completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time4 - time3)/3600, "hours")

graph(range(iterations+1), Fx, "f(x) vs iterations", "Iteration", "f(x)", "5di")

