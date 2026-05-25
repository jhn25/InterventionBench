from interventionbench.domains.base import DomainAdapter, RunResult

__all__ = ["DomainAdapter", "RunResult"]


def get_swebench_adapter(**kwargs):
    """Lazy import to avoid litellm/openai at import time."""
    from interventionbench.domains.swebench import SWEBenchAdapter

    return SWEBenchAdapter(**kwargs)


def get_osworld_adapter(**kwargs):
    """Lazy import — requires OSWorld venv (Python 3.10) via ~/osworld/.venv."""
    from interventionbench.domains.osworld import OSWorldAdapter

    return OSWorldAdapter(**kwargs)

