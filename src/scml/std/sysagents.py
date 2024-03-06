from scml.oneshot.sysagents import DefaultOneShotAdapter, _StdSystemAgent

__all__ = ["DefaultStdAdapter", "_StdSystemAgent"]


class DefaultStdAdapter(DefaultOneShotAdapter):
    """
    The base class of all agents running in Std based on StdAgent.

    Remarks:

        - It inherits from `Adapter` allowing it to just pass any calls not
          defined explicity in it to the internal `_obj` object representing
          the StdAgent.
    """
