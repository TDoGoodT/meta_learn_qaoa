import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Extended Rosenbrock function
def extended_rosenbrock(x):
    global eval_count
    eval_count += 1
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Gradient of the extended Rosenbrock function
def extended_rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    for i in range(1, len(x)-1):
        grad[i] = 200*(x[i]-x[i-1]**2) - 400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
    grad[-1] = 200*(x[-1]-x[-2]**2)
    return grad

def gradient_descent(func, grad, initial_guess, lr=0.0001, max_iters=5000):
    x = initial_guess
    for _ in range(max_iters):
        x = x - lr * grad(x)
        func(x)  # Count the evaluation
    return x

param_counts = list(range(2, 51))  # From 2 to 50 parameters
methods = ["COBYLA", "BFGS", "Nelder-Mead", "Powell"]
results = {}

for method in methods:
    evals = []
    for num_params in param_counts:
        # Reset evaluation counter
        eval_count = 0
        
        # Initial guess
        initial_guess = np.array([1.3] * num_params)
        
        # Perform the optimization
        minimize(extended_rosenbrock, initial_guess, method=method)
        
        # Store the number of evaluations
        evals.append(eval_count)
    
    results[method] = evals

# Plotting
plt.figure(figsize=(12, 7))
for method, evals in results.items():
    plt.plot(param_counts, evals, label=method)

plt.xlabel('Number of Parameters')
plt.ylabel('Number of Evaluations')
plt.title('Evaluations vs. Parameters for Different Optimization Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 7000)
plt.show()
