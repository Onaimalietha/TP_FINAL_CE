import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

class ControlSystem:
    def __init__(self, num, den, Kp, Ki, Kd, time):
        self.plant = ctrl.TransferFunction(num, den)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.pid_controller = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
        self.system = ctrl.feedback(self.pid_controller * self.plant, 1)
        self.time = time

    def run(self):
        t, response = ctrl.step_response(self.system, self.time)

        # Steady state error (final value error from desired value which is 1 in step response)
        steady_state_error = 1 - response[-1]

        # Overshoot
        overshoot = (max(response) - 1) * 100  # percentage overshoot

        # Settling time (time to remain within 2% of the final value)
        settling_time = None
        for i in range(len(response)):
            if all(np.abs(response[i:] - response[-1]) <= 0.02 * np.abs(response[-1])):
                settling_time = t[i]
                break

        # Performance metrics to minimize: steady state error, overshoot, settling time
        return steady_state_error, overshoot, settling_time