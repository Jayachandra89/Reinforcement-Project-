"""Simple baseline: change phase every fixed number of slots."""

def fixed_timer_action(step, interval=3):
    if interval < 1:
        raise ValueError('interval must be >= 1')
    return 1 if step > 0 and step % interval == 0 else 0
