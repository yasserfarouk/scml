import random

from negmas import ResponseType
from negmas.common import MechanismState
from negmas.sao import AspirationNegotiator
from typing import Optional

from scml.scml2019 import INVALID_UTILITY, CFP


class Mynegotiator(AspirationNegotiator):
    def __init__(self, name, ufun, cfp, partner_id):
        self.offers = {}  #  list of offers in that negotiator
        self.ufun = ufun
        self.partner_id = partner_id
        self.partner_offers = []
        self.cfp: CFP = cfp
        self.estimated_issue_weights = {}
        self.estimated_issue_values = {}
        self.n = 0.1
        issues = self.cfp.issues
        self.normalized_utilities = {}
        for issue in issues:
            self.estimated_issue_weights[issue.name] = 1 / len(issues)
            self.estimated_issue_values[issue.name] = {}
            for value in issue.values:
                self.estimated_issue_values[value] = 1 / len(issue.values)
        """"""
        super(Mynegotiator, self).__init__(name=name, ufun=ufun, partner_id=partner_id)

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        self.notify_ufun_changed()
        self._set_offers(offer)
        self.update_weights(offer)
        sacrafise_ratio = 0.02
        my_offer = self.propose(state=state)
        my_utility = self.get_normalized_utility(my_offer)
        offered_utility = self.get_normalized_utility(offer)
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )

        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        if offered_utility is None:
            return ResponseType.REJECT_OFFER
        if offered_utility != 0 and offered_utility >= my_utility:
            return ResponseType.ACCEPT_OFFER
        if offered_utility != 0 and (my_utility - sacrafise_ratio) <= offered_utility:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        self.notify_ufun_changed()
        util_lst = self._get_util()
        tot_util = len(util_lst)
        init_rat = round(tot_util * 0.10)  #  best 0,10 percent, random offers
        optm_rat = round(tot_util * 0.10)  # current_r + optm_r, increases the rank
        pesm_rat = round(tot_util * 0.05)  # current_r - pes_r, decreases the rank
        aspiration = self.aspiration(state.relative_time)
        asp = aspiration * (self.ufun_max - self.ufun_min) + self.ufun_min

        if not self.offers:
            return self._random_offer(init_rat)
        o = self.offers[self._ami.id]
        before_ix = len(o) - 1 if len(o) > 1 else len(o) - 2
        current_u = self.get_normalized_utility(
            o[-1]
        )  #  our utility from opponent offer
        cbefore_u = self.get_normalized_utility(o[before_ix])
        current_r = self._current_ranking(current_u, util_lst)
        return self._decide_offer(
            c_u=current_u,
            b_u=cbefore_u,
            c_r=current_r,
            optm_r=optm_rat,
            pesm_r=pesm_rat,
            t_util=tot_util,
            asp=asp,
            u_list=util_lst,
            init_r=init_rat,
        )

    def _get_util(self):
        return [order[0] for order in self.ordered_outcomes]

    def _set_offers(self, o):
        k = self._ami.id
        if k in self.offers:
            self.offers[k].append(o)
        else:
            self.offers[k] = [o]

    def _current_ranking(self, current_u, u_list):
        for i, u in enumerate(u_list):
            if u <= current_u:
                return i
        return round(len(u_list) / 2)  # median

    def _random_offer(self, init_r):
        return self.ordered_outcomes[random.randint(0, init_r)][1]

    def _decide_offer(self, c_u, b_u, c_r, optm_r, pesm_r, t_util, asp, u_list, init_r):
        if c_u < self.reserved_value:
            return self._random_offer(init_r)
        if asp < self.reserved_value:
            return self._random_offer(init_r)
        if c_u == b_u:
            return self._equality_offer(
                c_r=c_r, pesm_r=pesm_r, optm_r=optm_r, u_list=u_list, t_util=t_util
            )
        if c_u < b_u:
            return self._selfish_offer(c_r=c_r, optm_r=optm_r)
        if c_u > b_u:
            return self._generous_offer(
                c_r=c_r, pesm_r=pesm_r, u_list=u_list, t_util=t_util
            )

    def _selfish_offer(self, c_r, optm_r):
        # in case opponent is selfish to use, we propose selfish offers
        o = self.ordered_outcomes
        idx = c_r - optm_r
        return o[idx][1] if idx >= 0 else o[0][1]

    def _generous_offer(self, c_r, pesm_r, u_list, t_util):
        # in case opponent generous to us, we sacrifices from our utlitiy
        idx = c_r + pesm_r
        ord_out = self.ordered_outcomes
        r = self._current_ranking(self.reserved_value, u_list)
        if idx < t_util:
            ord_value = ord_out[idx]
            if ord_value[0] > self.reserved_value:
                return ord_value[1]
            else:
                return ord_out[r][1]
        else:
            return ord_out[r][1]

    def _equality_offer(self, c_r, pesm_r, optm_r, u_list, t_util):
        # in case opponent offers equal utility rate with the one before, we propose selfish offers
        return self._selfish_offer(c_r, optm_r)

    def on_leave(self, state: MechanismState) -> None:
        # print("ON LEAVE CALLED : "+self.name)
        super().on_leave(state)
        """"""

    def on_negotiation_end(self, state: MechanismState) -> None:
        # print("ON NEGOTIATION END CALLED : " + self.name)
        super().on_negotiation_end(state)

    def normalize(self):
        """"""
        isBiggestNegative = False
        if self.ordered_outcomes[0][0] < 0:
            isBiggestNegative = True
            max_utility = self.ordered_outcomes[0][0]
        else:
            max_utility = self.ordered_outcomes[0][0]
        for index in range(len(self.ordered_outcomes)):
            outcome = self.ordered_outcomes[index]
            if outcome[0] <= INVALID_UTILITY:
                self.ordered_outcomes[index] = (0, outcome[1])
            else:
                if isBiggestNegative:
                    self.ordered_outcomes[index] = (
                        (max_utility / outcome[0]),
                        outcome[1],
                    )
                else:
                    self.ordered_outcomes[index] = (
                        max(0, (outcome[0] / max_utility)),
                        outcome[1],
                    )
        a = 0

    def on_ufun_changed(self):
        super().on_ufun_changed()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
            reverse=True,
        )
        self.partner_offers.append(self.ordered_outcomes[0][1])
        self.update_weights(self.ordered_outcomes[1][1])
        if not self.assume_normalized:
            self.normalize()
            self.ufun_max = self.ordered_outcomes[0][0]
            self.ufun_min = self.ordered_outcomes[-1][0]
            if self.reserved_value is not None and self.ufun_min < self.reserved_value:
                self.ufun_min = self.reserved_value
        self.normalized_utilities = {}
        for offer_value_pair in self.ordered_outcomes:
            offer = offer_value_pair[1]
            value = offer_value_pair[0]
            self.normalized_utilities[str(offer)] = value

    def get_normalized_utility(self, offer):
        return self.normalized_utilities.get(str(offer))

    def update_weights(self, offer: "Outcome"):
        partner_offers = self.partner_offers
        if offer is not None:
            partner_offers.append(offer)
            if len(partner_offers) > 1:
                previous_offer = partner_offers[-2]
                for key in self.estimated_issue_weights.keys():
                    current_value = offer.get(key)
                    previous_value = previous_offer.get(key)
                    if current_value == previous_value:
                        self.estimated_issue_weights[key] = (
                            self.estimated_issue_weights.get(key) + self.n
                        )

            if len(partner_offers) > 1:
                weight_sum = 0
                for key in self.estimated_issue_weights.keys():
                    weight_sum += self.estimated_issue_weights.get(key)
                for key in self.estimated_issue_weights.keys():
                    self.estimated_issue_weights[key] = (
                        self.estimated_issue_weights.get(key) / weight_sum
                    )

            frequencies = {}
            # @ todo : Frequency heurstic

    def notify_ufun_changed(self):
        self.on_ufun_changed()

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        if self._ufun_modified:
            self.on_ufun_changed()
        result = self.propose(state=state)
        self.my_last_proposal = result
        return result

    def respond_(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self._ufun_modified:
            self.on_ufun_changed()
        return self.respond(state=state, offer=offer)
