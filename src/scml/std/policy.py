from scml.oneshot.policy import OneShotPolicy

__all__ = ["StdPolicy"]


class StdPolicy(OneShotPolicy):
    """
    A std agent structured in three components, state encoder, policy (action) and action decoder.

    The agent is divided into three components:

    1. State encoder (encode_state()) which takes the current state of all
       negotiation mechanisms, access the awi as needed, and generates a
       **state** which can be of any type to be passed to the next component.
    2. Policy (act()) which takes the state generated from the state encoder
       and returns an action which may be encoded as any type to be passed to
       the next component. *The policy (i.e. `act` () method) is not supposed
       to access the AWI or any other members of the class. It is preferred to
       be a pure function*. This makes it easy to test the policy at predefined
       conditions (i.e. states) without having to construct a simulation.
    3. Action decoder (decode_action()) which takes the action generated from
       the policy and generates the appropriate set of responses to all partners.

    Remarks:
        - The simplest form of state encoder which is implemented by default is
          to return the `state` member of the AWI.
        - The simplest form of action encoding is to simply return the responses as
          a `dict[str, SAOResponse]` from `act` which is then passed as it is by
          `decode_action` . This is the default implementation of `decode_action`
    """
