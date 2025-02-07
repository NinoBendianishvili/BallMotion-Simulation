import numpy as np
import time

def ball_trajectory(t, state):
    k, m, g = 0.01, 1, 9.81 
    x, y, vx, vy = state
    speed = np.sqrt(vx**2 + vy**2)
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -(k / m) * vx * speed
    dvy_dt = -g - (k / m) * vy * speed
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

class ODESolver:
    def __init__(self, f, t_span, y0, h):
        self.f = f  
        self.t_span = t_span
        self.y0 = y0  
        self.h = h  

    def euler_step(self, t, y):
        return y + self.h * self.f(t, y)

    def rk2_step(self, t, y):
        k1 = self.f(t, y)
        k2 = self.f(t + self.h / 2, y + self.h * k1 / 2)
        return y + self.h * k2

    def rk4_step(self, t, y):
        k1 = self.f(t, y)
        k2 = self.f(t + self.h / 2, y + self.h * k1 / 2)
        k3 = self.f(t + self.h / 2, y + self.h * k2 / 2)
        k4 = self.f(t + self.h, y + self.h * k3)
        return y + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(self, method="rk4"):
        methods = {"euler": self.euler_step, "rk2": self.rk2_step, "rk4": self.rk4_step}
        if method not in methods:
            raise ValueError(f"Method '{method}' not supported. Choose from 'euler', 'rk2', 'rk4'.")
        integration_method = methods[method]
        t_values = [self.t_span[0]]
        y_values = [self.y0]
        t = self.t_span[0]
        y = self.y0
        start_time = time.time()  

        while t < self.t_span[1]:
            y = integration_method(t, y)
            t += self.h
            t_values.append(t)
            y_values.append(y)

        elapsed_time = time.time() - start_time  
        if hasattr(self, "reference_solution"):
            reference_times, reference_values = self.reference_solution
            interpolated_reference = np.interp(t_values, reference_times, reference_values[:, 0])
            errors = np.linalg.norm(np.array(y_values)[:, 0] - interpolated_reference, axis=0)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            print(f"{method.upper()} method: max error = {max_error:.6f}, mean error = {mean_error:.6f}")

        print(f"{method.upper()} method took {elapsed_time:.6f} seconds for calculations.")
        return np.array(t_values), np.array(y_values)

    def set_reference_solution(self, reference_solution):
        """Set the reference solution for accuracy comparison."""
        self.reference_solution = reference_solution

if __name__ == "__main__":
    t_span = (0, 5)  
    y0 = [0, 0, 10, 10]  
    h = 0.01  
    solver = ODESolver(ball_trajectory, t_span, y0, h)

    print("Solving with RK4 method to obtain reference solution:")
    t_rk4, y_rk4 = solver.solve(method="rk4")
    solver.set_reference_solution((t_rk4, y_rk4))

    print("\nSolving with Euler method:")
    t_euler, y_euler = solver.solve(method="euler")

    print("\nSolving with RK2 method:")
    t_rk2, y_rk2 = solver.solve(method="rk2")

    print("\nSolving with RK4 method:")
    t_rk4, y_rk4 = solver.solve(method="rk4")
