import sympy as sp
import matplotlib.pyplot as plt
import os

x_1, x_2 = sp.symbols('x_1 x_2')

f = 100 * (x_2 - x_1**2)**2 + (1 - x_1)**2


def steepest_descent(x_k, gradient, hessian):
    # Returns the gradient at x_k
    return - gradient.subs({x_1: x_k[0], x_2: x_k[1]})


def newton(x_k, gradient, hessian):
    # Returns - H^-1 evaluted at x_k times the gradient at x_k
    return - hessian.subs({x_1: x_k[0], x_2: x_k[1]}).inv() * \
        gradient.subs({x_1: x_k[0], x_2: x_k[1]})


def graph(x, y, title, x_label, y_label, filename):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.ylabel(y_label)
    filename = "Tarea2" + os.sep + "images" + os.sep + filename + ".png"
    fig.savefig(filename, dpi=fig.dpi)
    plt.clf()


def minimize(f, x_0, method, a_0):
    rho = 0.5
    c = 0.5
    iterations = 0
    max_iterations = 20

    gradient = sp.Matrix([f.diff(x_1), f.diff(x_2)])
    hessian = sp.hessian(f, (x_1, x_2))

    if method == 'steepest_descent':
        p = steepest_descent
    elif method == 'newton':
        p = newton

    X = [x_0]
    F = [f.subs({x_1: x_0[0], x_2: x_0[1]})]
    A = [a_0]
    P = []

    P.append(p(X[-1], gradient, hessian))

    while iterations < max_iterations:
        while f.subs({x_1: X[-1][0] + A[-1] * P[-1][0],
                      x_2: X[-1][1] + A[-1] * P[-1][1]}) > f.subs({x_1: X[-1][0],
                                                                   x_2: X[-1][1]}) + c * A[-1] * gradient.subs({x_1: X[-1][0],
                                                                                                                x_2: X[-1][1]}).dot(P[-1]):
            A[-1] = rho * A[-1]

        X.append(X[-1] + A[-1] * P[-1])
        F.append(f.subs({x_1: X[-1][0], x_2: X[-1][1]}))
        P.append(p(X[-1], gradient, hessian))
        A.append(a_0)

        iterations += 1

    return X, F, A[:-1], P


X, F, A, P = minimize(f, x_0=sp.Matrix(
    [1.2, 1.2]), method='steepest_descent', a_0=1)

graph([i for i in range(len(A))],
      A,
      r"Step size, using Steepest Descent, $x_0 = [1.2, 1.2]$",
      "Iteration",
      r"$\alpha$",
      "steepest_descent_step_size_a")
graph([i for i in range(len(F))],
      F,
      r"$f(x)$, using Steepest Descent, $x_0 = [1.2, 1.2]$",
      "Iteration",
      r"$f(x)$",
      "steepest_descent_fx_a")

X, F, A, P = minimize(f, x_0=sp.Matrix([1.2, 1.2]), method='newton', a_0=1)

graph([i for i in range(len(A))],
      A,
      r"Step size, using Newton, $x_0 = [1.2, 1.2]$",
      "Iteration",
      r"$\alpha$",
      "newton_step_size_a")
graph([i for i in range(len(F))],
      F,
      r"$f(x)$, using Newton, $x_0 = [1.2, 1.2]$",
      "Iteration",
      r"$f(x)$",
      "newton_fx_a")

X, F, A, P = minimize(f, x_0=sp.Matrix(
    [-1.2, 1]), method='steepest_descent', a_0=1)

graph([i for i in range(len(A))],
      A,
      r"Step size, using Steepest Descent, $x_0 = [-1.2, 1]$",
      "Iteration",
      r"$\alpha$",
      "steepest_descent_step_size_b")
graph([i for i in range(len(F))],
      F,
      r"$f(x)$, using Steepest Descent, $x_0 = [-1.2, 1]$",
      "Iteration",
      r"$f(x)$",
      "steepest_descent_fx_b")

X, F, A, P = minimize(f, x_0=sp.Matrix([-1.2, 1]), method='newton', a_0=1)

graph([i for i in range(len(A))],
      A,
      r"Step size, using Newton, $x_0 = [-1.2, 1]$",
      "Iteration",
      r"$\alpha$",
      "newton_step_size_b")
graph([i for i in range(len(F))],
      F,
      r"$f(x)$, using Newton, $x_0 = [-1.2, 1]$",
      "Iteration",
      r"$f(x)$",
      "newton_fx_b")
