"""
Implements the base classes for all agents that can join a `SCML2020OneShotWorld`.


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
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from negmas import AgentMechanismInterface
from negmas import Contract
from negmas import Entity
from negmas import MechanismState
from negmas import Outcome
from negmas import PassThroughSAONegotiator
from negmas import ResponseType
from negmas import SAOController
from negmas import SAOResponse
from negmas import SAOSingleAgreementController
from negmas import SAOState
from negmas import SAOSyncController
from negmas.common import NegotiatorInfo
from negmas.helpers import get_class
from negmas.helpers import get_full_type_name
from negmas.outcomes import Issue
from negmas.sao import SAOAMI
from negmas.sao import SAONegotiator
from negmas.situated import RunningNegotiationInfo
from negmas.utilities import LinearUtilityAggregationFunction
from negmas.utilities import LinearUtilityFunction
from negmas.utilities import UtilityFunction
from negmas.utilities import normalize

__all__ = [
    "OneShotAgent",
    "OneShotSyncAgent",
    "OneShotSingleAgreementAgent",
    "OneShotIndNegotiatorsAgent",
    "EndingNegotiator",
]


class OneShotAgent(SAOController, Entity, ABC):
    """

    Base class for all agents in the One-Shot game.

    Args:
        owner: The adapter owning the agent. You do not need to directly deal
               with this.
        ufun: An optional `OneShotUFun` to set for the agent.
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

    def __init__(self, owner=None, ufun=None, name=None):
        super().__init__(
            default_negotiator_type=PassThroughSAONegotiator,
            default_negotiator_params=None,
            auto_kill=False,
            name=name,
            ufun=ufun,
        )
        self._awi = owner._awi if owner else None
        self._owner = owner
        self.ufun = owner.ufun if owner else None

    @property
    def awi(self):
        """Returns a `OneShotAWI` object for accessing the simulation."""
        return self._awi

    @property
    def running_negotiations(self) -> List[RunningNegotiationInfo]:
        """The negotiations currently requested by the agent.

        Returns:

            A list of negotiation information objects (`RunningNegotiationInfo`)
        """
        return self._owner.running_negotiations

    @property
    def unsigned_contracts(self) -> List[Contract]:
        """
        All contracts that are not yet signed.
        """
        return self._owner.unsigned_contracts

    def init(self):
        """
        Called once after the AWI is set.

        Remarks:
            - Use this for any proactive initialization code.
        """

    def make_ufun(self, add_exogenous=False):
        """
        Creates a utility function for the agent.

        Args:
            add_exogenous: If `True` then the exogenous contracts of the agent
                           will be automatically added whenever the ufun is
                           evaluated for any set of contracts, offers or otherwise.

        Remarks:
            - You can always as assume that self.ufun returns the ufun for your.
              You will not need to directly use this method in most cases.

        """
        return self._owner.make_ufun(add_exogenous)

    def step(self):
        """
        Called every step.

        Remarks:
            - Use this for any proactive code  that needs to be done every
              simulation step.
        """
        pass

    def connect_to_oneshot_adapter(self, owner):
        """Connects the agent to its adapter (used internally)"""
        self._owner = owner
        self._awi = owner._awi
        self.utility_function = owner.ufun

    def connect_to_2021_adapter(self, owner):
        """Connects the agent to its adapter (used internally)"""
        self._owner = owner
        self._awi = owner._awi
        self.utility_function = owner.ufun

    @abstractmethod
    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        """
        Proposes an offer to one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step

        Returns:
            an outcome to offer.
        """

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        """
        Responds to an offer from one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step
            offer: The offer received.

        Returns:
            A response type which can either be reject, accept, or end negotiation.

        Remarks:
            default behavior is to accept only if the current offer is the same
            or has a higher utility compared with what the agent would have
            proposed in the given state and reject otherwise

        """
        myoffer = self.propose(negotiator_id, state)
        if myoffer == offer:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    @property
    def internal_state(self) -> Dict[str, Any]:
        """
        Returns the internal state of the agent for debugging purposes.

        Remarks:
            - In your agent, you can add any key-value pair to this dict and
              then use agent_log_* methods to log this information at any point.
        """
        return {}

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """
        Called whenever a negotiation ends without agreement.

        Args:
            partners: List of the partner IDs consisting from self and the opponent.
            annotation: The annotation of the negotiation including the seller ID,
                        buyer ID, and the product.
            mechanism: The `AgentMechanismInterface` instance containing all information
                       about the negotiation.
            state: The final state of the negotiation of the type `SAOState`
                   including the agreement if any.
        """

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """
        Called whenever a negotiation ends with agreement.

        Args:
            contract: The `Contract` agreed upon.
            mechanism: The `AgentMechanismInterface` instance containing all information
                       about the negotiation that led to the `Contract` if any.
        """

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts (used internally)"""
        return [self.id] * len(contracts)

    def on_contract_executed(self, contract) -> None:
        pass

    def on_contract_breached(self, contract, breaches, resolution) -> None:
        pass

    def get_negotiator(self, partner_id: str) -> SAONegotiator:
        """
        Returns the negotiator corresponding to the given partner ID.

        Remarks:
            - Note that the negotiator ID and the partner ID are always the same.
        """
        return self.negotiators[partner_id][0]

    def get_ami(self, partner_id: str) -> SAOAMI:
        """
        Returns the `SAOAMI` (Agent Mechanism Interface) connecting the agent
        to the negotiation mechanism for the given partner.
        """
        return self.negotiators[partner_id][0].ami


class OneShotSyncAgent(SAOSyncController, OneShotAgent, ABC):
    """
    An agent that automatically accumulate offers from opponents and allows
    you to control all negotiations centrally in the `counter_all` method.

    Args:
        owner: The adapter owning the agent. You do not need to directly deal
               with this.
        ufun: An optional `OneShotUFun` to set for the agent.
        name: Agent name.

    """

    def __init__(self, *args, **kwargs):
        kwargs["global_ufun"] = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators
        (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Returns:
            A dictionary mapping negotiator ID to an `SAOResponse`. The response
            per agent consist of a tuple. In case of acceptance or ending the
            negotiation the second item of the tuple should be None. In case of
            rejection, the second item should be the counter offer.


        Remarks:
            - The response type CANNOT be WAIT.
            - If the system determines that a loop is formed, the agent may
            receive this call for a subset of negotiations not all of them.

        """

    @abstractmethod
    def first_proposals(self) -> Dict[str, "Outcome"]:
        """
        Gets a set of proposals to use for initializing the negotiation.

        Returns:
            A dictionary mapping each negotiator (in self.negotiators dict) to
            an outcome to be used as the first proposal if the agent is to start
            a negotiation.

        """
        return super().first_proposals()

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts (used internally)"""
        return [self.id] * len(contracts)


class OneShotSingleAgreementAgent(SAOSingleAgreementController, OneShotSyncAgent):
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

    def __init__(self, *args, strict: bool = False, **kwargs):
        super().__init__(*args, strict=strict, **kwargs)

    @abstractmethod
    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState) -> bool:
        """Should decide if the given offer is acceptable

        Args:
            offer: The offer being tested
            source: The ID of the negotiator that received this offer
            state: The state of the negotiation handled by that negotiator

        Remarks:
            - If True is returned, this offer will be accepted and all other
              negotiations will be ended.
        """

    @abstractmethod
    def best_offer(self, offers: Dict[str, "Outcome"]) -> Optional[str]:
        """
        Return the ID of the negotiator with the best offer

        Args:
            offers: A mapping from negotiator ID to the offer it received

        Returns:
            The ID of the negotiator with best offer. Ties should be broken.
            Return None only if there is no way to calculate the best offer.
        """

    @abstractmethod
    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        """Compares two outcomes of the same negotiation

        Args:
            a: "Outcome"
            b: "Outcome"
            negotiator: The negotiator for which the comparison is to be made
            state: Current state of the negotiation

        Returns:
            True if utility(a) > utility(b)
        """


class EndingNegotiator(SAONegotiator):
    def propose(self, state):
        return None

    def respond(self, state, offer):
        return ResponseType.END_NEGOTIATION


class OneShotIndNegotiatorsAgent(OneShotAgent):
    """
    A one-shot agent that deligates all of its decisions to a set of independent
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
            - All ufuns are of the type `LinearUtilityAggregationFunction`.
            - All ufuns are normalized with a maximum value of 1.0. You can
              use `normalize_ufuns=True` to gruarantee that.
            - All ufuns have a finite reserved value and at least one outcome is
             above it. You can guarantee that by using `set_reservation=True`.
            - All ufuns are created with `outcome_type=tuple`. See `test_ind_negotiators_genius()`
              at `tests/test_scml2021oneshot.py` for an example.
            - All weights of the `LinearUtilityAggregationFunction` must be between
              zero and one and the weights must sum to one.



    """

    def __init__(
        self,
        *args,
        default_negotiator_type="negmas.sao.AspirationNegotiator",
        default_negotiator_params=None,
        normalize_ufuns=False,
        set_reservation=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._default_negotiator_type = get_class(default_negotiator_type)
        self._default_negotiator_params = (
            dict() if not default_negotiator_params else default_negotiator_params
        )
        self._ufuns = dict()
        self._normalize = normalize_ufuns
        self._set_reservation = set_reservation

    @abstractmethod
    def generate_ufuns(self) -> Dict[str, UtilityFunction]:
        """
        Returns a utility function for each partner. All ufuns **MUST** be of
        type `LinearUtilityAggregationFunction` if a genius negotiator is used.
        """

    def generate_negotiator(self, partner_id: str) -> SAONegotiator:
        """
        Returns a negotiator to be used with some partner.

        Remarks:
            The default implementation will use the `default_negotiator_type`
            and `default_negotiator_params`.
        """
        return self._default_negotiator_type(**self._default_negotiator_params)

    def _urange(self, u: UtilityFunction, issues):
        if not isinstance(u, LinearUtilityAggregationFunction) and not isinstance(
            u, LinearUtilityFunction
        ):
            return u.utility_range(issues=issues)
        mn = mx = 0.0
        for (_, w), issue in zip(u.weights.items(), issues):
            values = list(issue.values)
            mnv, mxv = min(values), max(values)
            if w > 0:
                mn += mnv * w
                mx += mxv * w
            else:
                mn += mxv * w
                mx += mnv * w
        return mn, mx

    def _unorm(self, u: UtilityFunction, mn, mx):
        if not isinstance(u, LinearUtilityAggregationFunction) and not isinstance(
            u, LinearUtilityFunction
        ):
            return normalize(u, outcomes=Issue.enumerate(issues, max_n_outcomes=1000))
        # _, mx = self._urange(u, issues)
        if mx < 0:
            return None
        u.weights = {k: _ / mx for k, _ in u.weights.items()}
        return u

    def _get_ufuns(self):
        """
        Internam method that makes sure the reservation value is set to a
        meaningful value and that the ufun is normalized if needed
        """
        ufuns = self.generate_ufuns()
        if not self._normalize and not self._set_reservation:
            return ufuns
        for partner_id, u in ufuns.items():
            if self.awi.is_system(partner_id):
                continue
            issues = (
                self.awi.current_input_issues
                if partner_id in self.awi.my_suppliers
                else self.awi.current_output_issues
            )
            mn, mx = self._urange(u, issues)
            if self._normalize:
                u = self._unorm(u, mn, mx)
                if u is None:
                    continue
            if not self._set_reservation:
                continue
            if (
                u.reserved_value is None
                or u.reserved_value == float("-inf")
                or u.reserved_value == float("nan")
            ):
                u.reserved_value = mn - 1e-5
            u.reserved_value = u.reserved_value / mx
            if u.reserved_value > mx:
                ufuns[partner_id] = None
        return ufuns

    def init(self):
        super().init()
        self._ufuns = self._get_ufuns()

    def step(self):
        super().step()
        self._ufuns = self._get_ufuns()

    def make_negotiator(
        self,
        negotiator_type=None,
        name: str = None,
        **kwargs,
    ):
        """
        Creates a negotiator but does not add it to the controller. Call
        `add_negotiator` to add it.

        Args:
            negotiator_type: Type of the negotiator to be created.
            name: negotiator name
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        Returns:
            The negotiator to be controlled. None for failure

        Remarks:
            If you would like not to negotiate, just return `EndingNegotiator()`
            instead of None. The value None should only be returned if an exception
            is to be thrown.

        """
        ufun = self._ufuns[name]
        if ufun is None:
            return EndingNegotiator()
        negotiator = self.generate_negotiator(name)
        negotiator.id = name
        negotiator.name = name
        negotiator.ufun = ufun
        return negotiator

    def propose(self, negotiator_id, state):
        raise ValueError(
            "propose should never be called directly on OneShotIndNegotiatorsAgent"
        )
