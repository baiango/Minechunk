"""Walk/collision façade around TerrainRenderer collision methods."""

def update_camera_walk(renderer, dt: float) -> None:
    return renderer._update_camera_walk(dt)

def move_horizontal_with_step(renderer, position, axis: int, delta: float) -> bool:
    return renderer._move_horizontal_with_step(position, axis, delta)

def resolve_small_downward_snap(renderer, position, delta: float) -> bool:
    return renderer._resolve_small_downward_snap(position, delta)

def resolve_collision_axis(renderer, position, axis: int, delta: float) -> bool:
    return renderer._resolve_collision_axis(position, axis, delta)
