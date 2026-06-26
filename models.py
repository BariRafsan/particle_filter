from dataclasses import dataclass


@dataclass
class Ball:
    x:     float
    y:     float
    vx:    float
    vy:    float
    alive: bool = True


@dataclass
class Particle:
    x:      float
    y:      float
    vx:     float
    vy:     float
    weight: float = 1.0
