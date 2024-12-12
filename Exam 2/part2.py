import numpy as np
import pandas as pd
import time
from scipy.optimize import linprog

np.random.seed(3527)

# Problem dimensions
n = 5000
m = 2000

df = pd.read_csv('A.txt', header=None)
A = df.values  

# Generate c between -0.5 and 0.5
c = np.random.rand(n) - 0.5 

# Estimate f^* using scipy 
bounds = (-1, 1)
res = linprog(c, A_ub=A, b_ub=np.ones(m), bounds=[bounds]*n, method='highs')
f_star = res.fun
print("Estimated f* =", f_star)

#####################################################
# Helper functions
#####################################################

def project_onto_box(x, lower=-1.0, upper=1.0):
    return np.clip(x, lower, upper)

def feasibility_gap(x, A):
    # Measures how much constraints are violated:
    # max(a_j^T x - 1, 0) and max(|x_i|-1,0)
    Ax = A.dot(x)
    cons_viol = np.maximum(Ax - 1, 0).sum()
    box_viol = np.sum(np.maximum(np.abs(x)-1,0))
    return cons_viol + box_viol

def objective_value(x, c):
    return np.dot(c, x)

def penalty_function(x, c, A, mu=1000.0):
    # Unconstrained penalty form:
    # f(x) = <c,x> + mu/2 * sum_j max(a_j^T x -1, 0)^2 + mu/2 * sum_i max(|x_i|-1,0)^2
    Ax = A.dot(x)
    viol_linear = np.maximum(Ax-1,0)
    viol_box = np.maximum(np.abs(x)-1,0)
    return np.dot(c, x) + (mu/2.0)*(np.sum(viol_linear**2) + np.sum(viol_box**2))

def penalty_gradient(x, c, A, mu=1000.0):
    # Gradient of penalty_function
    Ax = A.dot(x)
    viol_linear = Ax - 1
    mask_linear = (viol_linear > 0)
    grad_linear = A[mask_linear,:].T.dot(viol_linear[mask_linear])  # sum over violating constraints
    
    diff_x = np.abs(x)-1
    mask_box_pos = (x > 1)
    mask_box_neg = (x < -1)
    # gradient w.r.t. box constraints
    # d/dx_i (max(|x_i|-1,0)^2) = 2(|x_i|-1)*sign(x_i) if violated
    grad_box = np.zeros_like(x)
    grad_box[mask_box_pos] = mu*(x[mask_box_pos]-1)
    grad_box[mask_box_neg] = mu*(x[mask_box_neg]+1)
    
    return c + mu*grad_linear + grad_box

def report_solution(x, f_star, A, c):
    obj = objective_value(x, c)
    opt_gap = obj - f_star
    feas_gap = feasibility_gap(x, A)
    return obj, opt_gap, feas_gap

#####################################################
# 1. Three methods for the linear problem
#####################################################

# We implement each method to run for a fixed amount of time and then report results.

def gradient_constant_step(x0, c, A, mu=1000.0, step=1e-5, max_time=10.0):
    x = x0.copy()
    start = time.time()
    while time.time() - start < max_time:
        grad = penalty_gradient(x, c, A, mu)
        x -= step*grad
    return x

def gradient_backtracking(x0, c, A, mu=1000.0, alpha_init=1e-3, beta=0.5, gamma=1e-4, max_time=10.0):
    x = x0.copy()
    start = time.time()
    while time.time() - start < max_time:
        grad = penalty_gradient(x, c, A, mu)
        fx = penalty_function(x, c, A, mu)
        step = alpha_init
        # Backtracking line search
        while True:
            x_new = x - step*grad
            f_new = penalty_function(x_new, c, A, mu)
            if f_new <= fx - gamma*step*np.dot(grad, grad):
                break
            step *= beta
            if step < 1e-20:
                break
        x = x_new
    return x

def nesterov_accelerated_backtracking(x0, c, A, mu=1000.0, alpha_init=1e-3, beta=0.5, gamma=1e-4, max_time=10.0):
    x = x0.copy()
    y = x0.copy()
    t_old = 1.0
    start = time.time()
    while time.time() - start < max_time:
        grad = penalty_gradient(y, c, A, mu)
        fy = penalty_function(y, c, A, mu)
        step = alpha_init
        # Backtracking line search
        while True:
            x_new = y - step*grad
            f_new = penalty_function(x_new, c, A, mu)
            if f_new <= fy - gamma*step*np.dot(grad, grad):
                break
            step *= beta
            if step < 1e-20:
                break
        x_prev = x
        x = x_new
        t_new = (1 + np.sqrt(1+4*t_old*t_old))/2
        y = x + (t_old-1)/t_new * (x - x_prev)
        t_old = t_new
    return x

# Run and compare
x0 = np.zeros(n)
time_budget = 10.0

x_const = gradient_constant_step(x0, c, A, mu=1000.0, step=1e-5, max_time=time_budget)
x_back = gradient_backtracking(x0, c, A, mu=1000.0, alpha_init=1e-3, max_time=time_budget)
x_nest = nesterov_accelerated_backtracking(x0, c, A, mu=1000.0, alpha_init=1e-3, max_time=time_budget)

print("Method A (Constant Step) Results:", report_solution(x_const, f_star, A, c))
print("Method B (Backtracking) Results:", report_solution(x_back, f_star, A, c))
print("Method C (Nesterov + Backtracking) Results:", report_solution(x_nest, f_star, A, c))

#####################################################
# 2. Barrier-type method
#####################################################

# Solve:
# min t <c,x> - sum_j log(1 - a_j^T x)
# subject to |x_i| <= 1
#
# We'll use gradient method with backtracking and projection to solve subproblems.
# Increase t iteratively.
# Stop when difference from f_star is small or feasibility gap is small.

def barrier_objective(x, c, A, t):
    Ax = A.dot(x)
    # If outside domain, return infinity
    if np.any(Ax >= 1):
        return np.inf
    return t*np.dot(c, x) - np.sum(np.log(1 - Ax))

def barrier_gradient(x, c, A, t):
    Ax = A.dot(x)
    denom = 1 - Ax
    # If outside domain, gradient is not defined properly
    if np.any(denom <= 0):
        return np.full_like(x, np.inf)
    grad = t*c + np.sum(A.T/(denom), axis=1)
    return grad

def solve_barrier_subproblem(x0, c, A, t, alpha_init=1e-3, beta=0.5, gamma=1e-4, max_iter=500, tol=1e-6):
    x = x0.copy()
    for _ in range(max_iter):
        grad = barrier_gradient(x, c, A, t)

        # If gradient is infinite, we are outside domain break 
        if np.any(np.isinf(grad)):
            break

        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break

        fx = barrier_objective(x, c, A, t)
        step = alpha_init

        # Add a maximum number of backtracking attempts
        max_backtracking = 50
        backtrack_count = 0
        success = False

        while backtrack_count < max_backtracking:
            x_new = project_onto_box(x - step*grad)
            f_new = barrier_objective(x_new, c, A, t)

            if f_new == np.inf:
                step *= beta
                backtrack_count += 1
                continue

            # Armijo condition
            if f_new <= fx - gamma * step * np.dot(grad, grad):
                success = True
                x = x_new
                break

            step *= beta
            backtrack_count += 1

        if not success:
            break

    return x

def barrier_method(c, A, x0=None, t0=1.0, mu=10.0, outer_max_iter=10, inner_tol=1e-6):
    if x0 is None:
        x0 = np.zeros_like(c)
    x = x0.copy()
    for k in range(outer_max_iter):
        x = solve_barrier_subproblem(x, c, A, t0, max_iter=1000, tol=inner_tol)
        obj = np.dot(c, x)
        feas = np.sum(np.maximum(A.dot(x)-1,0)) + np.sum(np.maximum(np.abs(x)-1,0))
        if feas < 1e-6:
            break
        t0 *= mu
    return x

# Run and compare
x_barrier = barrier_method(c, A, t0=1.0, mu=10.0, outer_max_iter=10, inner_tol=1e-6)
print("Barrier Method Results:", report_solution(x_barrier, f_star, A, c))
