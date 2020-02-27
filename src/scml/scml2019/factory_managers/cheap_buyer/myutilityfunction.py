import random

from negmas import UtilityValue, SAOState
from typing import Optional, Dict, Any

from scml.scml2019.common import SCMLAgreement, INVALID_UTILITY
from scml.scml2019.factory_managers.builtins import (
    AveragingNegotiatorUtility,
    GreedyFactoryManager,
)


class MyUtilityFunction(AveragingNegotiatorUtility):
    """"""

    def __init__(
        self,
        agent: GreedyFactoryManager,
        annotation: Dict[str, Any],
        ufun_id=-1,
        alpha=1,
        beta=0,
        average_number_of_steps=0,
    ):
        self.agent = agent
        self.annotation = annotation
        self.ufun_id = ufun_id
        self.alpha = alpha
        self.beta = beta
        self.average_number_of_steps = average_number_of_steps
        self.weight_of_u1 = 0
        self.weight_of_u2 = 1
        super(MyUtilityFunction, self).__init__(agent=agent, annotation=annotation)

    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        return self.get_utility_4(agreement=agreement)

    def get_utility_1(self, agreement):
        if len(self.agent.running_negotiations) == 0:
            self.optimism = 0
            return super().call(agreement)
        result = 0
        for negotiation in self.agent.running_negotiations:
            state: SAOState = negotiation.negotiator._ami.state
            self.optimism = state.relative_time
            result = result + super().call(agreement)
        return result / len(self.agent.running_negotiations)

    # def get_utility_monte_carlo(self, agreement):
    #     if self._free_sale(agreement):
    #         return INVALID_UTILITY
    #     result = 0
    #     results = []
    #     for i in range(self.number_of_simulations):
    #         hypothetical = [self._contract(agreement)]
    #         probability_of_acceptances = []
    #         for negotiation in self.agent.running_negotiations:
    #             negotiator = negotiation.negotiator
    #             current_offer = negotiator.my_last_proposal
    #             max_steps = negotiator._ami.n_steps
    #             step = negotiator._ami.state.step
    #             probability_of_acceptance = step/((self.alpha * max_steps) + (self.beta * self.average_number_of_steps))
    #             probability_of_acceptances.append(probability_of_acceptance)
    #             if current_offer is not None and random.random() < probability_of_acceptance:
    #                 hypothetical.append(self._contract(current_offer))
    #         base_util = self.agent.simulator.final_balance
    #         hypothetical_utility = self.agent.total_utility(hypothetical)
    #         if hypothetical_utility == float("-inf"):
    #
    #         print("HYPOTHETICAL UTILITY : ", hypothetical_utility)
    #         result = result + (hypothetical_utility - base_util)
    #         results.append(result)
    #     return result / self.number_of_simulations

    def get_utility_monte_carlo_2(self, agreement):
        if self._free_sale(agreement):
            return INVALID_UTILITY
        step_0_trimmed_running_negotiations = self.agent.running_negotiations
        index = 0
        while index < len(step_0_trimmed_running_negotiations):
            if (
                step_0_trimmed_running_negotiations[index].negotiator._ami.state.step
                == 0
            ):
                step_0_trimmed_running_negotiations.remove(
                    step_0_trimmed_running_negotiations[index]
                )
                index -= 1
            index += 1
        if len(step_0_trimmed_running_negotiations) > 0:
            percentage_utility_difference = float("inf")
            result = 0
            count = 0
            results = []
            average_utility = 0
            base_util = self.agent.simulator.final_balance
            while percentage_utility_difference > 0.01 and count < 10:
                hypothetical = [self._contract(agreement)]
                probability_of_acceptances = []
                for negotiation in step_0_trimmed_running_negotiations:
                    negotiator = negotiation.negotiator
                    current_offer = negotiator.my_last_proposal
                    max_steps = negotiator._ami.n_steps
                    step = negotiator._ami.state.step
                    probability_of_acceptance = step / (
                        (self.alpha * max_steps)
                        + (self.beta * self.average_number_of_steps)
                    )
                    probability_of_acceptances.append(probability_of_acceptance)
                    if (
                        current_offer is not None
                        and random.random() < probability_of_acceptance
                    ):
                        hypothetical.append(self._contract(current_offer))
                hypothetical_utility = self.agent.total_utility(hypothetical)
                u1 = hypothetical_utility - base_util
                u2 = self.agent.get_total_profit(hypothetical)
                resultant_utility = (u1 * self.weight_of_u1) + (u2 * self.weight_of_u2)
                result += resultant_utility
                count += 1
                average_utility = result / count
                results.append(resultant_utility)
                if count > 1:
                    percentage_utility_difference = abs(
                        (average_utility - resultant_utility) / average_utility
                    )
            return average_utility
        else:
            hypothetical = [self._contract(agreement)]
            base_util = self.agent.simulator.final_balance
            hypothetical = self.agent.total_utility(hypothetical)
            if hypothetical < 0:
                return float("-inf")
            return hypothetical - base_util

    def get_utility_expected(self, agreement):
        if self._free_sale(agreement):
            return INVALID_UTILITY

        step_0_trimmed_running_negotiations = self.agent.running_negotiations
        index = 0
        while index < len(step_0_trimmed_running_negotiations):
            if (
                step_0_trimmed_running_negotiations[index].negotiator._ami.state.step
                == 0
            ):
                step_0_trimmed_running_negotiations.remove(
                    step_0_trimmed_running_negotiations[index]
                )
                index -= 1
            index += 1

        hypothetical = [self._contract(agreement)]
        base_util = self.agent.simulator.final_balance
        count = 0
        total_utility = 0
        if len(step_0_trimmed_running_negotiations) > 0:
            for negotiation in step_0_trimmed_running_negotiations:
                negotiator = negotiation.negotiator
                current_offer = negotiator.my_last_proposal
                max_steps = negotiator._ami.n_steps
                step = negotiator._ami.state.step
                probability_of_acceptance = step / (
                    (self.alpha * max_steps)
                    + (self.beta * self.average_number_of_steps)
                )
                hypothetical_utility_with_reject_case = self.agent.total_utility(
                    hypothetical
                )
                if current_offer is not None:
                    hypothetical.append(self._contract(current_offer))
                    hypothetical_utility_with_accept_case = self.agent.total_utility(
                        hypothetical
                    )
                    weighted_hypothetical_utility = (
                        probability_of_acceptance
                        * hypothetical_utility_with_accept_case
                        + (1 - probability_of_acceptance)
                        * hypothetical_utility_with_reject_case
                    )
                    weighted_utility = weighted_hypothetical_utility - base_util
                else:
                    weighted_utility = hypothetical_utility_with_reject_case - base_util
                total_utility += weighted_utility
                count += 1
            return total_utility / count
        else:
            hypothetical = [self._contract(agreement)]
            base_util = self.agent.simulator.final_balance
            hypothetical = self.agent.total_utility(hypothetical)
            if hypothetical < 0:
                return float("-inf")
            return hypothetical - base_util

    def get_utility_4(self, agreement):
        quantity = agreement.get("quantity")
        unit_price = agreement.get("unit_price")
        unit_cost = self.agent.get_average_buying_price()
        return max((unit_price - unit_cost) * quantity, 0)

    # def get_utility_3(self, agreement):
    #     if self._free_sale(agreement):
    #         return INVALID_UTILITY
    #     awi = self.agent.awi
    #     joint_probability = 0
    #     binary_combinations = list(itertools.product([0, 1], repeat=len(self.agent.running_negotiations)))
    #     for negotiation in self.agent.running_negotiations:
    #         state = negotiation.negotiator._ami.state
