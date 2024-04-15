import attr
from scml.oneshot.agents.rand import EqualDistOneShotAgent, RandDistOneShotAgent
from scml.oneshot.agents.greedy import GreedyOneShotAgent, GreedySingleAgreementAgent
from scml.utils import anac2024_oneshot

# required for typing
from typing import Any

# required for development
from scml.oneshot import OneShotSyncAgent

# required for typing
from negmas import SAONMI, Contract, Outcome, SAOResponse, SAOState, ResponseType


class AWITester(OneShotSyncAgent):
    def first_proposals(self) -> dict[str, Outcome | None]:
        nmis = self.awi.current_nmis
        return dict(zip(nmis.keys(), [n.random_outcome() for n in nmis.values()]))

    def counter_all(  # type: ignore
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        partners = self.awi.my_suppliers + self.awi.my_consumers
        nmis = self.awi.current_nmis
        assert self.awi.settings
        n_steps = self.awi.settings["neg_n_steps"]
        n_steps2 = [_.n_steps for _ in self.awi.current_nmis.values()][0]
        assert n_steps == n_steps2

        steps = {k: v.step for k, v in states.items()}
        assert len(offers) == len(
            self.awi.current_offers
        ), f"{offers=}\n{self.awi.current_offers=}"
        for k, v in offers.items():
            assert (
                v == self.awi.current_offers[k]
            ), f"{k=}: {offers[k]=}, {self.awi.current_offers[k]=}"
        assert len(states) == len(
            self.awi.current_states
        ), f"{states=}\n{self.awi.current_states=}"
        for k, v in states.items():
            assert (
                v.step == self.awi.current_states[k].step
            ), f"{k=}: {states[k]=}, {self.awi.current_states[k]=}"

        buy = {k for k in self.awi.current_negotiation_details["buy"].keys()}
        sell = {k for k in self.awi.current_negotiation_details["sell"].keys()}
        mynegs = []
        commonset = (
            set(offers.keys())
            .intersection(set(states.keys()))
            .intersection(set(self.awi.current_nmis.keys()))
            .intersection(set(self.awi.current_states.keys()))
            .intersection(set(self.awi.current_offers.keys()))
        )
        for n in self.awi._world._negotiations.values():
            pids = [_.id for _ in n.partners]
            partner = [_ for _ in pids if _ != self.id][0]
            if self.id in pids and partner not in commonset:
                mynegs.append(
                    (
                        pids,
                        n.mechanism._current_state if n.mechanism else None,
                        n.annotation["sim_step"],
                    )
                )
        assert not (
            set(self.awi.current_nmis.keys()) != set(offers.keys())
            or set(states.keys()) != set(offers.keys())
            or set(self.awi.current_states.keys()) != set(offers.keys())
            or set(self.awi.current_offers.keys()) != set(offers.keys())
        ), (
            f"{self.id} Erred at step {self.awi.current_step}\n"
            f"{self.awi.my_suppliers}\n{self.awi.my_consumers}\n"
            f"Outcomes:{offers}\n"
            f"Offers:{list(offers.keys())}\nStates:{list(states.keys())}\n"
            f"AWI states:{list(self.awi.current_states.keys())}\n"
            f"AWI nmis:{list(self.awi.current_nmis.keys())}\n"
            f"AWI nmis:{list(nmis.keys())}\n"
            f"AWI offerers:{list(self.awi.current_offers.keys())}\n"
            f"AWI offers:{list(self.awi.current_offers.values())}\n"
            f"Partners:{partners}\n"
            f"Steps:{steps}\n"
            f"Annotations: {[_.annotation for _ in self.running_negotiations]}\n"
            f"Buy:{buy}\n"
            f"Sell:{sell}\n"
            f"My Negs:{mynegs}\n"
            f"Ended:{self.failed}\n"
            f"Succeeded:{self.succeeded}\n"
        )

        return dict(
            zip(
                offers.keys(),
                [
                    SAOResponse(
                        ResponseType.REJECT_OFFER,
                        nmis[_].random_outcome() if _ in nmis else None,
                    )
                    for _ in offers
                ],
            )
        )

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        self.failed, self.succeeded = set(), set()

    def step(self):
        """Called at at the END of every production step (day)"""
        self.failed, self.succeeded = set(), set()

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: SAONMI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""
        partner = [_ for _ in partners if _ != self.id][0]
        self.failed.add(
            (
                partner,
                str(attr.asdict(state)),
                tuple(mechanism._mechanism.extended_trace),  # type: ignore
                annotation["sim_step"],
            )
        )

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""
        partner = [_ for _ in contract.partners if _ != self.id][0]
        self.succeeded.add(
            (
                partner,
                str(attr.asdict(mechanism.state)),
                tuple(contract.agreement.values()),
                mechanism.annotation["sim_step"],
            )
        )


def test_awi_nmis_states_offers_match_in_counter_all():
    competitors = [
        AWITester,
        RandDistOneShotAgent,
        EqualDistOneShotAgent,
        GreedySingleAgreementAgent,
        GreedyOneShotAgent,
    ]

    anac2024_oneshot(
        competitors=competitors,
        verbose=True,
        n_steps=5,
        n_configs=2,
        neg_hidden_time_limit=float("inf"),
        neg_time_limit=float("inf"),
        neg_step_time_limit=float("inf"),
        debug=True,
        # parallelism="serial",
        ignore_agent_exceptions=False,
        ignore_negotiation_exceptions=False,
        ignore_simulation_exceptions=False,
        ignore_contract_execution_exceptions=False,
        safe_stats_monitoring=False,
    )
