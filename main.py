from simulation      import BallSimulation
from particle_filter import ParticleFilter
from visualization   import Visualizer
from config          import NUM_BALLS, TOTAL_TIME, NUM_PARTICLES


def main():
    print("=" * 55)
    print("  Particle Filter  —  Portfolio Exam 2")
    print("=" * 55)
    print(f"  Balls      : {NUM_BALLS}")
    print(f"  Sim time   : {TOTAL_TIME} s")
    print(f"  Particles  : {NUM_PARTICLES}")
    print("=" * 55)

    print("\n[1/3] Simulating ball trajectories ...")
    sim = BallSimulation()
    sim.run()

    print("[2/3] Initializing particle filter ...")
    pf = ParticleFilter()

    print("[3/3] Starting animation  (close window to quit) ...\n")
    viz = Visualizer(sim, pf)
    viz.animate()


if __name__ == "__main__":
    main()
