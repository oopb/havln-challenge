"""Short forward motions use the fast current-depth probe, not terrain analysis."""


def forward_probe(geometry, angle=0.):
    probe = getattr(geometry, 'movement_probe', None)
    return probe(angle) if probe is not None else geometry.assess_step(angle)


def check_action(action, geometry):
    if action == 'MOVE_FORWARD':
        check = forward_probe(geometry)
        if check['state'] != 'EXECUTABLE':
            return None, check['reason']
    return action, None
