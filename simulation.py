import numpy as np

from models import Ball
from config import (
    NUM_BALLS, MIN_START_X, MAX_START_X, MIN_START_Y, MAX_START_Y,
    MIN_SPEED, MAX_SPEED, MIN_ANGLE, MAX_ANGLE,
    DT, TOTAL_TIME, GRAVITY, GROUND_LEVEL,
    POSITION_NOISE_STD, OBSERVATION_INTERVAL, DROP_START, DROP_END,
)


class BallSimulation:

    def __init__(self):
        self.balls = []
        self.true_history        = []
        self.observation_history = []
        self._create_balls()

    def _create_balls(self):
        for _ in range(NUM_BALLS):
            x     = np.random.uniform(MIN_START_X, MAX_START_X)
            y     = np.random.uniform(MIN_START_Y, MAX_START_Y)
            speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
            angle = np.deg2rad(np.random.uniform(MIN_ANGLE, MAX_ANGLE))
            vx    = speed * np.cos(angle)
            vy    = speed * np.sin(angle)
            self.balls.append(Ball(x=x, y=y, vx=vx, vy=vy))

    def _step_ball(self, ball):
        if not ball.alive:
            return
        ball.x  += ball.vx * DT
        ball.y  += ball.vy * DT
        ball.vy -= GRAVITY * DT

        if ball.y <= GROUND_LEVEL:
            ball.y     = GROUND_LEVEL
            ball.alive = False

    def _observe(self, step_index):
        if DROP_START <= step_index <= DROP_END:
            return []
        if step_index % OBSERVATION_INTERVAL != 0:
            return []

        observations = []
        for ball in self.balls:
            if not ball.alive:
                continue
            obs_x = ball.x + np.random.normal(0, POSITION_NOISE_STD)
            obs_y = ball.y + np.random.normal(0, POSITION_NOISE_STD)
            observations.append(np.array([obs_x, obs_y]))
        return observations

    def run(self):
        for step in range(int(TOTAL_TIME / DT)):
            for ball in self.balls:
                self._step_ball(ball)

            self.true_history.append([
                np.array([b.x, b.y, b.vx, b.vy])
                for b in self.balls if b.alive
            ])
            self.observation_history.append(self._observe(step))
