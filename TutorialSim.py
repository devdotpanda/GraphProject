import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------
# Simulation Constants
# -----------------------
plt.style.use('dark_background')

DT = 1e-19        # timestep in seconds
SIM_LEN = 2000    # number of integration steps
EPS = 1e-12       # epsilon for numerical stability
COULOMB_K = 2.533e38  # Coulomb constant in MeV·pm³/(e²·s²)

# Particle properties
m = np.array([938.0, 0.511, 938.0])  # Mass of each particle (MeV)
q = np.array([1.0, -1.0, 1.0])       # Charge of each particle (in e)

# -----------------------
# Initial State Setup
# -----------------------
state0 = np.zeros((3, 4), dtype=float)  # columns: x, y, vx, vy

dist = 52.9  # pm (approx hydrogen orbit)
state0[1, 0] = dist * np.cos(1.1)  # electron start x
state0[1, 1] = dist * np.sin(1.1)  # electron start y

# Initial electron velocity (perpendicular to radius)
vi = np.sqrt(np.abs(COULOMB_K * q[1] * q[0]) / (dist * m[1] + EPS))
dx, dy = state0[1, 0] - state0[0, 0], state0[1, 1] - state0[0, 1]
r = np.hypot(dx, dy) + EPS
state0[1, 2] = vi * (dy / r)
state0[1, 3] = -vi * (dx / r)

# Second proton
state0[2, 0] = -190.0
state0[2, 2] = vi * 0.4

# Shift scene slightly left
state0[:, 0] -= 20.0

bound = dist * 3.0  # field plotting range

# -----------------------
# Physics Functions
# -----------------------
def coulomb_accel(xi, yi, xj, yj, qi, qj, mi):
    dx = xj - xi
    dy = yj - yi
    r2 = dx**2 + dy**2 + EPS
    r = np.sqrt(r2)
    factor = COULOMB_K * qi * qj / (mi * (r2 * r + EPS))
    return factor * dx, factor * dy


def get_derivative(y_flat):
    """Compute the derivative of the state vector (for RK4)."""
    state = np.reshape(y_flat, (-1, 4))
    n = state.shape[0]
    d = np.zeros_like(state)

    # velocity derivatives
    d[:, 0] = state[:, 2]
    d[:, 1] = state[:, 3]

    # acceleration derivatives
    for i in range(n):
        xi, yi = state[i, 0], state[i, 1]
        for j in range(n):
            if i == j:
                continue
            xj, yj = state[j, 0], state[j, 1]
            ax, ay = coulomb_accel(xi, yi, xj, yj, q[i], q[j], m[i])
            d[i, 2] += ax
            d[i, 3] += ay

    return d.flatten()


def simulate_steps(state0, h, steps):
    """RK4 fixed-step integrator."""
    y = state0.flatten()
    trajectory = [np.reshape(y, state0.shape).copy()]
    for _ in range(steps):
        k1 = get_derivative(y)
        k2 = get_derivative(y + 0.5 * h * k1)
        k3 = get_derivative(y + 0.5 * h * k2)
        k4 = get_derivative(y + h * k3)
        y += (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(np.reshape(y, state0.shape).copy())
    return np.array(trajectory)

# -----------------------
# Electric Field Function
# -----------------------
E_PLOT_N = 100
xg = np.linspace(-bound, bound, E_PLOT_N)
yg = np.linspace(-bound, bound, E_PLOT_N)
X, Y = np.meshgrid(xg, yg)

def E_field(state):
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    for i in range(state.shape[0]):
        xi, yi = state[i, 0], state[i, 1]
        rx, ry = X - xi, Y - yi
        r2 = rx**2 + ry**2 + EPS
        denom = (r2)**1.5
        Ex += COULOMB_K * q[i] * rx / (denom + EPS)
        Ey += COULOMB_K * q[i] * ry / (denom + EPS)
    return Ex, Ey

# -----------------------
# Run Simulation
# -----------------------
print("Running simulation...")
simulation = simulate_steps(state0, DT, SIM_LEN)
print("Simulation complete.")

# -----------------------
# Visualization
# -----------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.set_aspect('equal')
ax.set_title("Final Particle Positions and Electric Field")

# compute final field
Ex, Ey = E_field(simulation[-1])
E_strength = np.log(Ex**2 + Ey**2 + EPS)
mesh = ax.pcolormesh(X, Y, E_strength, shading='auto', cmap='inferno')

# particle scatter
scatter = ax.scatter(
    simulation[-1][:, 0], simulation[-1][:, 1],
    s=np.log(m / np.min(m) + 1) * 40,
    c=q, cmap='seismic', vmin=-2, vmax=2, edgecolors='white'
)

plt.show()
