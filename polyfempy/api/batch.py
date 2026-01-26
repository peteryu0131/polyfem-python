from .solve import solve

def batch_solve(jobs):
    """Sequential batch solve (order-preserving).

    Args:
        jobs: Sequence of (V, C, cfg) or (V, C, cfg, kwargs_dict) tuples.

    Returns:
        list[Result] in same order as jobs.
    """
    out = []
    for job in jobs:
        if len(job) == 3:
            V, C, cfg = job
            kwargs = {}
        else:
            V, C, cfg, kwargs = job
            if kwargs is None:
                kwargs = {}
        res = solve(V, C, cfg, **kwargs)
        out.append(res)
    return out
