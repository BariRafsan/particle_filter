import copy
import numpy as np
from sklearn.cluster import DBSCAN

from models import Particle
from config import (
    NUM_PARTICLES,
    MIN_START_X, MAX_START_X, MIN_START_Y, WORLD_HEIGHT,
    MIN_SPEED, MAX_SPEED, MIN_ANGLE, MAX_ANGLE,
    DT, GRAVITY, GROUND_LEVEL,
    POSITION_NOISE_STD, PROCESS_NOISE_POSITION, PROCESS_NOISE_VELOCITY,
    RESAMPLE_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
)


class ParticleFilter:

    def __init__(self):
        self.particles = []
        self._init_particles()

    def _init_particles(self):
        uniform_weight = 1.0 / NUM_PARTICLES
        for _ in range(NUM_PARTICLES):
            x     = np.random.uniform(MIN_START_X, MAX_START_X)
            y     = np.random.uniform(MIN_START_Y, WORLD_HEIGHT)
            speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
            angle = np.deg2rad(np.random.uniform(MIN_ANGLE, MAX_ANGLE))
            vx    = speed * np.cos(angle)
            vy    = speed * np.sin(angle)
            self.particles.append(
                Particle(x=x, y=y, vx=vx, vy=vy, weight=uniform_weight)
            )

    def predict(self):
        for p in self.particles:
            p.x  += p.vx * DT + np.random.normal(0, PROCESS_NOISE_POSITION)
            p.y  += p.vy * DT + np.random.normal(0, PROCESS_NOISE_POSITION)
            p.vx +=             np.random.normal(0, PROCESS_NOISE_VELOCITY)
            p.vy += -GRAVITY * DT + np.random.normal(0, PROCESS_NOISE_VELOCITY)
            if p.y < GROUND_LEVEL:
                p.y = GROUND_LEVEL

    @staticmethod
    def _gaussian_likelihood(distance):
        sigma = POSITION_NOISE_STD
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))

    def update(self, observations):
        if not observations:
            return

        for particle in self.particles:
            # Score against the closest observation (max-match rule).
            # Needed because observations cannot be assigned to specific balls.
            best_likelihood = 0.0
            for obs in observations:
                dist = np.sqrt((particle.x - obs[0]) ** 2 +
                               (particle.y - obs[1]) ** 2)
                likelihood = self._gaussian_likelihood(dist)
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
            particle.weight *= best_likelihood

    def normalize_weights(self):
        total = sum(p.weight for p in self.particles)
        if total < 1e-300:
            uniform = 1.0 / NUM_PARTICLES
            for p in self.particles:
                p.weight = uniform
            return
        for p in self.particles:
            p.weight /= total

    def effective_sample_size(self):
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights ** 2)

    def systematic_resample(self):
        cumulative = np.cumsum([p.weight for p in self.particles])
        step       = 1.0 / NUM_PARTICLES
        start      = np.random.uniform(0, step)

        indices = []
        i = 0
        for m in range(NUM_PARTICLES):
            u = start + m * step
            while cumulative[i] < u:
                i += 1
            indices.append(i)

        uniform = 1.0 / NUM_PARTICLES
        new_particles = []
        for idx in indices:
            p        = copy.copy(self.particles[idx])
            p.weight = uniform
            new_particles.append(p)
        self.particles = new_particles

    def step(self, observations):
        self.predict()
        self.update(observations)
        self.normalize_weights()
        if self.effective_sample_size() < RESAMPLE_THRESHOLD:
            self.systematic_resample()

    def estimate(self):
        positions  = np.array([[p.x, p.y] for p in self.particles])
        weights    = np.array([p.weight    for p in self.particles])
        labels     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(positions)

        estimates = []
        for label in np.unique(labels):
            if label == -1:
                continue

            mask    = labels == label
            w       = weights[mask]
            w_total = w.sum()
            if w_total == 0:
                continue

            cluster = [self.particles[i] for i, m in enumerate(mask) if m]
            w_norm  = w / w_total

            estimates.append({
                'x':  sum(w_norm[j] * cluster[j].x  for j in range(len(cluster))),
                'y':  sum(w_norm[j] * cluster[j].y  for j in range(len(cluster))),
                'vx': sum(w_norm[j] * cluster[j].vx for j in range(len(cluster))),
                'vy': sum(w_norm[j] * cluster[j].vy for j in range(len(cluster))),
            })

        return estimates

    def particle_positions(self):
        return np.array([[p.x, p.y] for p in self.particles])

    def particle_weights(self):
        return np.array([p.weight for p in self.particles])
