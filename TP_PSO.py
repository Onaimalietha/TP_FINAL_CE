import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from system import *
import time

desired_error = 0
desired_overshoot = 5
desired_ts = 2

Ke = 1
Ko = 1
Kt = 1

# PSO tryout
def f(x):
    for params in x:
        Kp, Ki, Kd = params
        control_system = ControlSystem(num, den, Kp, Ki, Kd, t)
        steady_state_error, overshoot, settling_time = control_system.run()
        error_dif = abs(steady_state_error - desired_error)
        overshoot_dif = abs(overshoot - desired_overshoot)
        time_dif = abs(settling_time - desired_ts)
        # Calculate a weighted sum of the performance metrics
        f = (Ke*error_dif + Ko*overshoot_dif + Kt*time_dif)
    return f

# Plant parameters
num = [1]
den = [1, 2, 1]
t = np.linspace(0, 10, 2500)

# Create bounds
# kp ki kd
max_bound = [20, 10, 10]
min_bound = [0, 0, 0]
bounds = (np.array(min_bound), np.array(max_bound))

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=3, options=options, bounds=bounds)

start = time.time()
# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)
finish = time.time()

print(f"Optimized PID parameters: Kp = {pos[0]:.4f}, Ki = {pos[1]:.4f}, Kd = {pos[2]:.4f}")


# Test the optimized PID controller
optimized_system = ControlSystem(num, den, pos[0], pos[1], pos[2], t)
steady_state_error, overshoot, settling_time = optimized_system.run()

print(f"Steady State Error: {steady_state_error:.4f}")
print(f"Overshoot: {overshoot:.2f}%")
print(f"Settling Time: {settling_time:.2f} seconds")
print("Processing time:", finish - start)
    

# Plot the step response of the optimized system
t, response = ctrl.step_response(optimized_system.system, t)
plt.figure(figsize=(10, 6))
plt.plot(t, response, label='Optimized Step Response')
plt.axhline(1, color='r', linestyle='--', label='Desired Value')
plt.title('Step Response with Optimized PID Controller')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid(True)
plt.show()
