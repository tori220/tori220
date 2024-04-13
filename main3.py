import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parameters
a = 100  # Thermal diffusivity
length = 50  # mm, domain size
time = 4  # seconds, total simulation time
nodes = 40  # number of nodes in each dimension

# Initialization
dx = length / nodes  # spatial step in x-direction
dy = length / nodes  # spatial step in y-direction
dt = 0.3 * dx * dy / a  # time step, chosen for stability
t_nodes = int(time / dt)  # number of time steps

u = np.zeros((nodes, nodes)) + 20  # Initial temperature in degrees C, uniform

# Boundary conditions
u[:, 0] = u[:, -1] = u[0, :] = u[-1, :] = 100  # Setting edges to 100°C

# Setup plot for animation
fig, ax = plt.subplots()
pcm = ax.imshow(u, cmap='jet', vmin=0, vmax=100)
plt.colorbar(pcm, ax=ax)

def update(frame):
    global u
    w = u.copy()  # Work on a copy to avoid overwriting values prematurely
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            # Calculate second derivatives using central difference
            dd_ux = (w[i-1, j] - 2 * w[i, j] + w[i+1, j]) / dx**2
            dd_uy = (w[i, j-1] - 2 * w[i, j] + w[i, j+1]) / dy**2

            # Update temperature using discretized diffusion equation
            u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]

    # Update plot
    pcm.set_data(u)
    ax.set_title(f"Distribution at t: {frame * dt:.3f} [s]")
    return pcm,

# Create animation
ani = FuncAnimation(fig, update, frames=range(t_nodes), blit=True)

# To display the animation inline in a Jupyter Notebook, you would typically use
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# If you are not in a notebook, or you want to save the animation to a file, you can do so like this:
# ani.save('heat_distribution.mp4', fps=30)  # Save as MP4
plt.show()  # Show the plot

