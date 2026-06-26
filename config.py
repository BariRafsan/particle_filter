import numpy as np

# --- World ---
WORLD_WIDTH  = 50.0
WORLD_HEIGHT = 50.0
GROUND_LEVEL = 0.0
GRAVITY      = 9.81   # m/s^2

DT         = 0.05   # seconds per step
TOTAL_TIME = 8.0    # seconds

# --- Balls ---
NUM_BALLS   = 3

MIN_START_X = 0.0
MAX_START_X = 50.0
MIN_START_Y = 0.0
MAX_START_Y = 5.0

MIN_SPEED = 12.0
MAX_SPEED = 30.0

MIN_ANGLE = 20   # degrees
MAX_ANGLE = 75

# --- Sensor ---
POSITION_NOISE_STD   = 1.5   # meters, std-dev of Gaussian measurement noise
OBSERVATION_INTERVAL = 3     # produce a reading every N steps
DROP_START           = 50    # step where sensor dropout begins
DROP_END             = 90    # step where sensor dropout ends

# --- Particle Filter ---
NUM_PARTICLES          = 4000
PROCESS_NOISE_POSITION = 0.15
PROCESS_NOISE_VELOCITY = 0.40
RESAMPLE_THRESHOLD     = NUM_PARTICLES * 0.5

# --- DBSCAN clustering ---
DBSCAN_EPS         = 2.5
DBSCAN_MIN_SAMPLES = 10

SEED = 42
np.random.seed(SEED)
