# engine/optimizer/pso/velocity_update.py

import random

def update_velocity(velocity, position, pbest, gbest, w=0.5, c1=1.5, c2=1.5):
    """
    Update the velocity vector using the PSO update rule.

    Args:
        velocity (list): Current velocity.
        position (list): Current position.
        pbest (list): Personal best position.
        gbest (list): Global best position.
        w (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.

    Returns:
        list: Updated velocity vector.
    """
    new_velocity = []
    for i in range(len(position)):
        r1 = random.random()
        r2 = random.random()
        v_new = (
                w * velocity[i] +
                c1 * r1 * (pbest[i] - position[i]) +
                c2 * r2 * (gbest[i] - position[i])
        )
        new_velocity.append(v_new)

    return new_velocity
