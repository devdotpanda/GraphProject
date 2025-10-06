import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

# Setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Parameters
pointsNum = 30
radius = 1

# Calculate all point coordinates
angles = np.linspace(0, 2 * np.pi, pointsNum, endpoint=False)
points = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

# Generate all line pairs (i, j)
connections = [(i, j) for i in range(pointsNum) for j in range(pointsNum)]

# Create empty line objects (initially invisible)
lines = []
for _ in connections:
    (line,) = ax.plot([], [], lw=1, color='cyan', alpha=0.7)
    lines.append(line)

# Set bounds
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

def update(frame):
    """Draw one more line each frame."""
    # Draw up to the current frame
    for k in range(frame + 1):
        i, j = connections[k]
        x1, y1 = points[i]
        x2, y2 = points[j]
        lines[k].set_data([x1, x2], [y1, y2])
    return lines

# Animation — runs until all lines are drawn
ani = FuncAnimation(
    fig,
    update,
    frames=len(connections),
    interval=14,  # speed (ms per frame)
    blit=True,
    repeat=False
)

plt.show()
