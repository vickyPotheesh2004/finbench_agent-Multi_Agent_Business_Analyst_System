"""
src/utils/system_utils.py
"""

try:
    import resource
except ImportError:
    resource = None


def get_memory_usage_mb():

    if resource is None:
        return 0.0

    try:

        return (
            resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / 1024
        )

    except Exception:

        return 0.0