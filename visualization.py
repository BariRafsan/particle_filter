import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import WORLD_WIDTH, WORLD_HEIGHT, GROUND_LEVEL, DROP_START, DROP_END


class Visualizer:

    def __init__(self, simulation, particle_filter):
        self.sim = simulation
        self.pf  = particle_filter

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.suptitle("Particle Filter — Ball Tracking", fontsize=14, fontweight="bold")

    def animate(self):
        total_steps = len(self.sim.observation_history)
        self._anim = FuncAnimation(
            self.fig, self._draw,
            frames=total_steps, interval=40, repeat=False,
        )
        plt.tight_layout()
        plt.show()

    def _draw(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-2, WORLD_WIDTH + 2)
        self.ax.set_ylim(-1, WORLD_HEIGHT + 2)
        self.ax.set_xlabel("x  (meters)")
        self.ax.set_ylabel("y  (meters)")
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=GROUND_LEVEL, color="black", linewidth=2, label="Ground")

        observations = self.sim.observation_history[frame]
        truth        = self.sim.true_history[frame]

        self.pf.step(observations)

        positions = self.pf.particle_positions()
        self.ax.scatter(positions[:, 0], positions[:, 1],
                        s=3, c="gray", alpha=0.2, label="Particles", zorder=1)

        if truth:
            self.ax.scatter([b[0] for b in truth], [b[1] for b in truth],
                            s=120, c="green", marker="o", zorder=4, label="True position")

        if observations:
            self.ax.scatter([o[0] for o in observations], [o[1] for o in observations],
                            s=200, c="royalblue", marker="+", linewidths=2,
                            zorder=3, label="Observation")

        estimates = self.pf.estimate()
        for est in estimates:
            self.ax.scatter(est["x"], est["y"], s=250, c="red", marker="X", zorder=5)
            scale = 0.4
            self.ax.annotate(
                "",
                xy=(est["x"] + est["vx"] * scale, est["y"] + est["vy"] * scale),
                xytext=(est["x"], est["y"]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.8),
                zorder=5,
            )

        if estimates:
            self.ax.scatter([], [], s=200, c="red", marker="X", label="Estimate")

        sensor_status = "DROPOUT" if DROP_START <= frame <= DROP_END else "OK"
        ess           = self.pf.effective_sample_size()
        self.ax.set_title(
            f"Step: {frame:4d}   Sensor: {sensor_status}   "
            f"ESS: {ess:.0f}   Balls detected: {len(estimates)}",
            fontsize=10,
        )
        self.ax.legend(loc="upper right", fontsize=9)
