from .solve import solve

def batch_solve(jobs):
    """
    Sequential batch solve (order-preserving)

    Parameters
    ----------
    jobs : sequence
        Each item is either:
          - a 3-tuple (V, C, cfg), or
          - a 4-tuple (V, C, cfg, kwargs_dict)
        where kwargs_dict contains keyword args forwarded to solve(), e.g.
        sidesets_func / dtype.

    Returns
    -------
    list[Result]
        A list of results in exactly the same order as the input jobs.
    """
    out = []
    for job in jobs:
        if len(job) == 3:
            V, C, cfg = job
            kwargs = {}
        else:
            V, C, cfg, kwargs = job
            kwargs = kwargs or {}
        res = solve(V, C, cfg, **kwargs)
        out.append(res)
    return out
