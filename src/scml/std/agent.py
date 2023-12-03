"""
Implements the base classes for all agents that can join a `SCML2024StdWorld`.


Remarks:
    - You can access all of the negotiators associated with the agent using
      `self.negotiators` which is a dictionary mapping the `negotiator_id` to
      a tuple of two values: The `SAONegotiator` object and a key-value context
      dictionary. In 2021, the context will always be empty.
    - The `negotiator_id` associated with a negotiation with some partner will
      be the same as the agent ID of that partner. This means that all negotiators
      engaged with some partner over all simulation steps will have the same ID
      which is useful if you are keeping information about past negotiations and
      partner behavior.
"""

from scml.oneshot.agent import (
    EndingNegotiator,
    OneShotAgent,
    OneShotIndNegotiatorsAgent,
    OneShotSingleAgreementAgent,
    OneShotSyncAgent,
)

__all__ = [
    "StdAgent",
    "StdSyncAgent",
    "StdSingleAgreementAgent",
    "StdIndNegotiatorsAgent",
    "EndingNegotiator",
]


class StdAgent(OneShotAgent):
    """

    Base class for all agents in the std game.

    Args:
        owner: The adapter owning the agent. You do not need to directly deal
               with this.
        ufun: An optional `StdUFun` to set for the agent.
        name: Agent name.

    Remarks:
        - You can access all of the negotiators associated with the agent using
          `self.negotiators` which is a dictionary mapping the `negotiator_id` to
          a tuple of two values: The `SAONegotiator` object and a key-value context
          dictionary. In 2021, the context will always be empty.
        - The `negotiator_id` associated with a negotiation with some partner will
          be the same as the agent ID of that partner. This means that all negotiators
          engaged with some partner over all simulation steps will have the same ID
          which is useful if you are keeping information about past negotiations and
          partner behavior.
    """


class StdSyncAgent(OneShotSyncAgent):
    """
    An agent that automatically accumulate offers from opponents and allows
    you to control all negotiations centrally in the `counter_all` method.

    Args:
        owner: The adapter owning the agent. You do not need to directly deal
               with this.
        ufun: An optional `StdUFun` to set for the agent.
        name: Agent name.

    """


class StdSingleAgreementAgent(OneShotSingleAgreementAgent):
    """
    A synchronized agent that tries to get no more than one agreement.

    This controller manages a set of negotiations from which only a single one
    -- at most -- is likely to result in an agreement.
    To guarantee a single agreement, pass `strict=True`

    The general algorithm for this controller is something like this:

        - Receive offers from all partners.
        - Find the best offer among them by calling the abstract `best_offer`
          method.
        - Check if this best offer is acceptable using the abstract `is_acceptable`
          method.

            - If the best offer is acceptable, accept it and end all other negotiations.
            - If the best offer is still not acceptable, then all offers are rejected
              and with the partner who sent it receiving the result of `best_outcome`
              while the rest of the partners receive the result of `make_outcome`.

        - The default behavior of `best_outcome` is to return the outcome with
          maximum utility.
        - The default behavior of `make_outcome` is to return the best offer
          received in this round if it is valid for the respective negotiation
          and the result of `best_outcome` otherwise.

    Args:
        strict: If True the controller is **guaranteed** to get a single
                agreement but it will have to send no-response repeatedly so
                there is a higher chance of never getting an agreement when
                two of those controllers negotiate with each other
    """


class StdIndNegotiatorsAgent(OneShotIndNegotiatorsAgent):
    """
    A std agent that deligates all of its decisions to a set of independent
    negotiators (one per partner per day).

    Args:
        default_negotiator_type: An `SAONegotiator` descendent to be used for
                                 creating all negotiators. It can be passed either
                                 as a class object or a string with the full class
                                 name (e.g. "negmas.sao.AspirationNegotiator").
        default_negotiator_type: A dict specifying the paratmers used to create
                                 negotiators.
        normalize_ufuns: If true, all utility functions will be normalized to have
                         a maximum of 1.0 (the minimum value may be negative).
        set_reservation: If given, the reserved value of all ufuns will be
                         guaranteed to be between the minimum and maximum of
                         the ufun. This is needed to avoid failures of some
                         GeniusNegotiators.

    Remarks:

        - To use this class, you need to override `generate_ufuns`. If you
          want to change the negotiator type used depending on the partner, you
          can also override `generate_negotiator`.
        - If you are using a `GeniusNegotiator` you must guarantee the following:
            - All ufuns are of the type `LinearAdditiveUtilityFunction`.
            - All ufuns are normalized with a maximum value of 1.0. You can
              use `normalize_ufuns=True` to gruarantee that.
            - All ufuns have a finite reserved value and at least one outcome is
             above it. You can guarantee that by using `set_reservation=True`.
            - All weights of the `LinearAdditiveUtilityFunction` must be between
              zero and one and the weights must sum to one.



    """
