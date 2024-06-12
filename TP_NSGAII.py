import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
import time
from system import *

desired_error = 0
desired_overshoot = 2
desired_ts = 0.5

class PIDOptimizationProblem(ElementwiseProblem):
    def __init__(self, num, den, time):
        super().__init__(n_var=3, n_obj=3, n_constr=0, xl=[-20, 0, 0], xu=[20, 10, 10])
        self.num = num
        self.den = den
        self.time = time

    def _evaluate(self, x, out, *args, **kwargs):
        Kp, Ki, Kd = x
        control_system = ControlSystem(self.num, self.den, Kp, Ki, Kd, self.time)
        steady_state_error, overshoot, settling_time = control_system.run()
        error_dif = abs(steady_state_error - desired_error)
        overshoot_dif = abs(overshoot - desired_overshoot)
        time_dif = abs(settling_time - desired_ts)
        out["F"] = [error_dif, overshoot_dif, time_dif]

# Plant parameters
num = [1]
den = [1, 2, 1]
t = np.linspace(0, 10, 1000)

# Define the problem
problem = PIDOptimizationProblem(num, den, t)

# Define the algorithm
algorithm = NSGA2(pop_size=100)

# Define the termination criterion
termination = get_termination("n_gen", 50)

start = time.time()
# Perform the optimization
res = minimize(problem, algorithm, termination, seed=1, verbose=True)
finish = time.time()
# Extract the optimized results
opt_params = res.X
opt_performance = res.F

# Print and plot the results
print("Optimized PID parameters:")
for i, params in enumerate(opt_params):
    Kp, Ki, Kd = params
    print(f"Solution {i+1}: Kp = {Kp:.4f}, Ki = {Ki:.4f}, Kd = {Kd:.4f}")

# Visualize the trade-offs
plt.figure(figsize=(10, 6))
plt.scatter(opt_performance[:, 0], opt_performance[:, 1], c=opt_performance[:, 2], cmap='viridis')
plt.colorbar(label='Settling Time (s)')
plt.xlabel('Steady State Error')
plt.ylabel('Overshoot (%)')
plt.title('Pareto Front')
plt.grid(True)
plt.show()

# Test and plot the best solution found
best_solution_idx = np.argmin(np.sum(opt_performance, axis=1))  # Minimize sum of objectives
best_params = opt_params[best_solution_idx]
best_system = ControlSystem(num, den, best_params[0], best_params[1], best_params[2], t)
steady_state_error, overshoot, settling_time = best_system.run()

print(f"Best Solution PID parameters: Kp = {best_params[0]:.4f}, Ki = {best_params[1]:.4f}, Kd = {best_params[2]:.4f}")
print(f"Steady State Error: {steady_state_error:.4f}")
print(f"Overshoot: {overshoot:.2f}%")
print(f"Settling Time: {settling_time:.2f} seconds")
print("Processing time:", finish - start)

# Plot the step response of the best solution
t, response = ctrl.step_response(best_system.system, t)
plt.figure(figsize=(10, 6))
plt.plot(t, response, label='Best Step Response')
plt.axhline(1, color='r', linestyle='--', label='Desired Value')
plt.title('Step Response with Optimized PID Controller')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid(True)
plt.show()
