import numpy as np


def get_seen_boids(boid_id, boids, perception):
    """Get array of boids closer than perception,
    with an extra dimension containing squared distances."""
    boid = boids[boid_id]
    other_boids = boids  # np.delete(boids, boid_id, axis=0)
    dists_squared = (boid[0] - other_boids[:, 0]) ** 2 + (boid[1] - other_boids[:, 1]) ** 2
    seen_boids = other_boids[dists_squared < perception**2]
    seen_boids = np.concatenate(
        (seen_boids, np.expand_dims(dists_squared[dists_squared < perception**2], 1)), axis=1
    )
    return seen_boids


def get_seen_predators(boid, predators, perception):
    """Get array of predators closer than perception,
    with an extra dimension containing squared distances."""
    dists_squared = (boid[0] - predators[:, 0]) ** 2 + (boid[1] - predators[:, 1]) ** 2
    seen_predators = predators[dists_squared < perception**2]
    seen_predators = np.concatenate(
        (seen_predators, np.expand_dims(dists_squared[dists_squared < perception**2], 1)), axis=1
    )
    return seen_predators


def avoid_predators(boid, predators, avoid_factor):
    acc = 0.5 * (boid[:2] - predators[:, :2].mean(axis=0))
    acc += 0.5 * (boid[2:4] - predators[:, 2:4].mean(axis=0))
    return avoid_factor * acc


def avoid_edges(boids, margin, width, height, avoid_factor):
    """Turn around when closer than margin to an edge."""
    acc = np.zeros((len(boids), 2))
    # commented parts correspond to an avoidance acc proportional to distance to border
    acc[:, 0] += boids[:, 0] < margin  # * (margin - boids[:, 0])
    acc[:, 0] -= boids[:, 0] > (width - margin)  # * (boids[:, 0] - width + margin)
    acc[:, 1] += boids[:, 1] < margin  # * (margin - boids[:, 1])
    acc[:, 1] -= boids[:, 1] > (height - margin)  # * (boids[:, 1] - height + margin)
    return avoid_factor * acc


def cohesion(boid, seen_boids, cohesion_factor):
    """Cohesion rule, boids get closer to center of mass of seen boids."""
    acc = seen_boids[:, :2].mean(axis=0) - boid[:2]
    return cohesion_factor * acc


def separation(boid, seen_boids, separation_factor, safe_space):
    """Seperation rule, boids avoid other boids that are closer than safe_space."""
    acc = boid[:2] - seen_boids[:, :2]
    # acc /= np.stack((seen_boids[:, -1],) * 2, 1) # uncomment for seperation acc inversly proportional to squared distance
    acc[seen_boids[:, -1] > safe_space**2] = 0
    return separation_factor * acc.sum(axis=0)


def alignment(boid, seen_boids, alignment_factor):
    """Alignment rule, boids align their speed with seen boids."""
    acc = seen_boids[:, 2:4].mean(axis=0) - boid[2:4]
    return alignment_factor * acc


def update_boids(
    boids,
    perception,
    predators,
    pred_perception,
    safe_space,
    max_speed,
    cohesion_factor,
    separation_factor,
    alignment_factor,
    avoid_factor,
    avoid_pred_factor,
    margin,
    width,
    height,
):
    """Update boids' position and velocity.

    Args:
        boids (np.ndarray): array of shape (n_boids, 4), with the 4 dims of axis 1 being x, y, dx and dy in that order.
        perception (int): Max distance at which a boid perceives an other boid.
        safe_space (int): A boid will try to avoid boids that are closer than this distance (seperation rule).
        max_speed (float): Maximum speed, speed norms are capped at that value.
        cohesion_factor (float): Scaling factor to the cohesion rule.
        separation_factor (float): Scaling factor to the seperation rule.
        alignment_factor (float): Scaling factor to the alignment rule.
        avoid_factor (float): Scaling factor to the edge avoidance.
        margin (int): If a boid is closer than this distance from a border, it will turn around.
        width (int): width of window.
        height (int): height of window.

    Returns:
        np.ndarray: updated boids.

    """
    total_acc = avoid_edges(boids, margin, width, height, avoid_factor)
    for i, boid in enumerate(boids):
        seen_boids = get_seen_boids(i, boids, perception)
        if len(seen_boids):
            total_acc[i] += cohesion(boid, seen_boids, cohesion_factor)
            total_acc[i] += separation(boid, seen_boids, separation_factor, safe_space)
            total_acc[i] += alignment(boid, seen_boids, alignment_factor)
        if len(predators):
            seen_predators = get_seen_predators(boid, predators, pred_perception)
            if len(seen_predators):
                total_acc[i] += avoid_predators(boid, seen_predators, avoid_pred_factor)
        boids[i, 2:] += total_acc[i]
        if boids[i, 2] ** 2 + boids[i, 3] ** 2 > max_speed**2:
            boids[i, 2:] = max_speed * boids[i, 2:] / np.sqrt(boids[i, 2] ** 2 + boids[i, 3] ** 2)
        boids[i, :2] += boids[i, 2:]
    return boids
