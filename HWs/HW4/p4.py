import matplotlib.pyplot as plt
import os
from time import perf_counter
from numpy import float16, dot, ones, zeros, identity, log, outer, sqrt
from numpy.linalg import norm, solve, cond
import pandas as pd
from progress.bar import Bar
from random import sample

n = 5000
m = 2000
epsilon = float16(1e-6)
# C is the vector of ones
ONE = float16(1)
TWO = float16(2)
rho = float16(0.75)

f_optimal = -1264.24311

max_iterations = 20 # Maximum number of iterations
max_sub_iterations = 5 # Maximum number of sub-iterations

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

    result = ones(n)
    with Bar('Gradient', fill='#', suffix='%(percent).1f%% - %(eta)ds', max=n) as bar:
        for i in range(n):
            for j in range(m):
                result[i] += A[j][i] / (ONE - optimized_dot(A[j],x))
            result[i] += TWO * x[i] / (ONE - x[i]*x[i])
            bar.next()

    gradients[stringx] = result
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

## quasi-Newton BFGS with backtracking and wolfe conditions

time0 = perf_counter()


alpha = float16(0.15) # Initial alpha
c_1 = float16(10e-4)
c_2 = float16(0.9)
X = [zeros(n, dtype=float16)] # Initial point x_0 = 0
Fx = [f(X[-1],A)]
GRADIENTS = [gradient(X[-1], A)]
ALPHAS = [alpha]
H = [identity(n)]
P = [-H[-1] @ GRADIENTS[-1]]
H_condition_numbers = [cond(H[-1])]

iterations = 0


while (norm(GRADIENTS[-1]) > epsilon) and (iterations < max_iterations):
    print("Iteration", iterations)
    print("Condition number of H_k:", H_condition_numbers[-1])
    print("f(x_k):", Fx[-1])
    print("x_k:", X[-1]) 

    iterations += 1
    sub_iterations = 0
    
    while (f(X[-1] + ALPHAS[-1] * P[-1], A) > f(X[-1], A) + c_1 * ALPHAS[-1] * optimized_dot(gradient(X[-1], A), P[-1]) 
           or (- optimized_dot(gradient(X[-1] + ALPHAS[-1] * P[-1], A), P[-1]) > - c_2 * optimized_dot(gradient(X[-1], A), P[-1]))
           ) and sub_iterations < max_sub_iterations:
        sub_iterations += 1
        ALPHAS[-1] = rho * ALPHAS[-1]

    X.append(X[-1] + ALPHAS[-1] * P[-1]) # X_k+1 = X_k + alpha * P_k
    Fx.append(f(X[-1], A))
    GRADIENTS.append(gradient(X[-1], A))
    s = X[-1] - X[-2]
    y = GRADIENTS[-1] - GRADIENTS[-2]
    ALPHAS.append(alpha)
    H.append(H[-1] - (H[-1]@ (outer(y,y) @H[-1]))/(y.T @(H[-1]@y)) + outer(s,s)/(optimized_dot(y,s)))
    P.append(- H[-1] @ GRADIENTS[-1])
    H_condition_numbers.append(cond(H[-1]))

time1 = perf_counter()

print("Quasi-Newton BFGS with backtracking and wolfe conditions completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time1 - time0)/3600, "hours")

log_Fx = [log(f - f_optimal) for f in Fx]
graph(range(iterations+1), log_Fx, "log(f(x)-f*) vs iterations", "Iteration", "log(f(x)-f*)", "4ai")
graph(range(iterations+1), H_condition_numbers, "Condition number of H vs iterations", "Iteration", "Condition number of H", "4aii")



## Confidence region Newton method 

dots = {}
fs = {}
gradients = {}
zero_vector = zeros(n)

def dogleg(grad, hess, rad):
    p_B = -solve(hess, grad)
    p_U = - optimized_dot(grad, grad) / (grad.T @ hess @ grad) * grad

    if norm(p_B) <= rad:
        return p_B
    elif norm(p_U) >= rad:
        return rad/norm(p_U) * p_U
    else:
        a = optimized_dot(p_B, p_B) - 2 * optimized_dot(p_B, p_U) + optimized_dot(p_U, p_U)
        b = 2 * optimized_dot(p_B, p_U) - 2 * optimized_dot(p_U, p_U)
        c = optimized_dot(p_U, p_U) - rad**2
        tau_minus_one = (-b + (b**2 - 4*a*c)**0.5) / (2*a)
        return p_U + tau_minus_one * (p_B - p_U)
    
def m_function_generator(fx, hess, grad):
    return lambda p: fx + optimized_dot(grad, p) + 0.5 * p.T @ hess @ p

        
def compute_rho(f, x, p, A, m):
    return (f(x, A) - f(x + p, A)) / (m(zero_vector)-m(p))


time2 = perf_counter()
X = [zeros(n, dtype=float16)] # Initial point x_0 = 0
Fx = [f(X[-1],A)]
DELTAS = [0.05] # Radius of the confidence region
Delta_max = 0.25
H = [identity(n)] # Initial Hessian
GRADIENTS = [gradient(X[-1], A)] # Initial gradient
eta = 0.25 # threshold
P = [dogleg(GRADIENTS[-1], H[-1], DELTAS[-1])] # Initial step
M = [m_function_generator(f(X[-1], A), H[-1], GRADIENTS[-1])] # Initial m function
Rho = [compute_rho(f, X[-1], P[-1], A, M[-1])] # Initial rho

iterations = 0

while (norm(GRADIENTS[-1]) > epsilon) and (iterations < max_iterations):
    print("Iteration", iterations)
    print("f(x_k):", f(X[-1],A))
    print("x_k:", X[-1]) 
    print("Delta_k:", DELTAS[-1])

    iterations += 1
    sub_iterations = 0

    if Rho[-1] < 0.25:
        DELTAS.append(DELTAS[-1] / 4)
    else:
        if Rho[-1] > 0.75 and norm(P[-1]) == DELTAS[-1]:
            DELTAS.append(min(2*DELTAS[-1], Delta_max))
        else:
            DELTAS.append(DELTAS[-1])
    
    if Rho[-1] > eta:
        X.append(X[-1] + P[-1])
    else:
        X.append(X[-1]) 

    # Delta and X were updated, now we update the other

    Fx.append(f(X[-1], A))
    GRADIENTS.append(gradient(X[-1], A))

    s = X[-1] - X[-2] #Update H by BFGS formula
    y = GRADIENTS[-1] - GRADIENTS[-2]
    H.append(H[-1] - (H[-1]@ (outer(y,y) @H[-1]))/(y.T @(H[-1]@y)) + outer(s,s)/(optimized_dot(y,s)))

    P.append(dogleg(GRADIENTS[-1], H[-1], DELTAS[-1]))
    M.append(m_function_generator(f(X[-1], A), H[-1], GRADIENTS[-1]))
    Rho.append(compute_rho(f, X[-1], P[-1], A, M[-1]))

time3 = perf_counter()

print("Confidence region Newton method completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time3 - time2)/3600, "hours")

log_Fx = [log(f - f_optimal) for f in Fx]
graph(range(iterations+1), log_Fx, "log(f(x)-f*) vs iterations", "Iteration", "log(f(x)-f*)", "4bi")
graph(range(iterations+1), DELTAS, "Delta vs iterations", "Iteration", "Delta", "4bii")


## Accelerated Nesterov method with backtracking

dots = {}
fs = {}
gradients = {}
time4 = perf_counter()

X = [zeros(n, dtype=float16)] # Initial point x_0 = 0
Fx = [f(X[-1],A)]
lambda_prev = 0
lambda_curr = 1
gamma = 1
y_prev = X[-1]
alpha = 0.15 # Initial alpha
GRADIENTS = [gradient(X[-1], A)] # Initial gradient

iterations = 0

while (norm(GRADIENTS[-1]) > epsilon) and (iterations < max_iterations):
    print("Iteration", iterations)
    print("f(x_k):", Fx[-1])
    print("x_k:", X[-1]) 

    iterations += 1
    sub_iterations = 0

    y_curr = X[-1] - alpha * GRADIENTS[-1]
    X.append((ONE - gamma) * y_curr + gamma * y_prev)
    y_prev = y_curr

    lambda_tmp = lambda_curr
    lambda_curr = (1 + sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
    lambda_prev = lambda_tmp

    gamma = (1 - lambda_prev) / lambda_curr

    Fx.append(f(X[-1], A))
    GRADIENTS.append(gradient(X[-1], A))

time5 = perf_counter()

print("Accelerated Nesterov method with backtracking completed")
print("Iterations:", iterations)
print("Total time elapsed:", (time5 - time4)/3600, "hours")

log_Fx = [log(f - f_optimal) for f in Fx]
graph(range(iterations+1), log_Fx, "log(f(x)-f*) vs iterations", "Iteration", "log(f(x)-f*)", "4ci")

## Stochastic gradient descent with mini-batches of size m/4, m/10, m/20

def stochastic_gradient(x, A, batch):
    result = ones(n)
    with Bar('Gradient', fill='#', suffix='%(percent).1f%% - %(eta)ds', max=n) as bar:
        for i in range(n):
            for j in batch:
                result[i] += A[j][i] / (ONE - optimized_dot(A[j],x))
            result[i] += TWO * x[i] / (ONE - x[i]*x[i])
            bar.next()
    return result

def stochastic_gradient_method(batch_size):
    time_a = perf_counter()
    alpha = 0.15 # Initial alpha
    X = [zeros(n, dtype=float16)] # Initial point x_0 = 0
    Fx = [f(X[-1],A)]
    batch = sample(range(m), batch_size)
    batch.sort()
    STOCHASTIC_GRADIENTS = [stochastic_gradient(X[-1], A, batch)] # Initial gradient
    iterations = 0
    while (norm(STOCHASTIC_GRADIENTS[-1]) > epsilon) and (iterations < max_iterations):
        print("Iteration", iterations)
        print("f(x_k):", Fx[-1])
        print("x_k:", X[-1])
        print("Batch size:", batch_size)
        iterations += 1
        X.append(X[-1] - alpha * STOCHASTIC_GRADIENTS[-1])
        Fx.append(f(X[-1], A))
        batch = sample(range(m), batch_size)
        batch.sort()
        STOCHASTIC_GRADIENTS.append(stochastic_gradient(X[-1], A, batch))
    time_b = perf_counter()
    print("Stochastic gradient method with batch size", batch_size, "completed")
    print("Iterations:", iterations)
    print("Total time elapsed:", (time_b - time_a)/3600, "hours")
    return Fx

dots = {}
fs = {}
batch_size = m//4 # Size of the mini-batches
Fx = stochastic_gradient_method(batch_size)
log_Fx = [log(f - f_optimal) for f in Fx]
graph(range(iterations+1), log_Fx, "log(f(x)-f*) vs iterations", "Iteration", "log(f(x)-f*)", "4di")

dots = {}
fs = {}
batch_size = m//10 # Size of the mini-batches
Fx = stochastic_gradient_method(batch_size)
log_Fx = [log(f - f_optimal) for f in Fx]
graph(range(iterations+1), log_Fx, "log(f(x)-f*) vs iterations", "Iteration", "log(f(x)-f*)", "4dii")

dots = {}
fs = {}
batch_size = m//20 # Size of the mini-batches
Fx = stochastic_gradient_method(batch_size)
log_Fx = [log(f - f_optimal) for f in Fx]
graph(range(iterations+1), log_Fx, "log(f(x)-f*) vs iterations", "Iteration", "log(f(x)-f*)", "4diii")