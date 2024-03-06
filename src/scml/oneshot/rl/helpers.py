"""Helpers for the observation and action managers."""

import numpy as np
from collections import defaultdict
from typing import Mapping, TypeVar
from scml.oneshot.awi import OneShotAWI
from negmas.outcomes import Outcome
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE, OneShotState
from scml.oneshot.rl.common import group_partners


__all__ = [
    "recover_offers",
    "read_offers",
    "encode_offers_with_time",
    "encode_offers_no_time",
    "decode_offers_no_time",
    "unnormalize_offers",
    "normalize_offers_no_time",
    "clip_normal",
    "clip",
    "discretize_and_clip",
    "normalize_and_clip",
]


def recover_offers(
    encoded: np.ndarray,
    awi: OneShotState | OneShotAWI,
    n_suppliers: int,
    n_consumers: int,
    max_group_size: int,
    continuous: bool,
    n_prices: int,
) -> dict[str, Outcome | None]:
    suppliers = group_partners(awi.my_suppliers, n_suppliers, max_group_size)
    consumers = group_partners(awi.my_consumers, n_consumers, max_group_size)
    buyos = awi.current_input_outcome_space
    sellos = awi.current_output_outcome_space
    return decode_offers_no_time(
        encoded,
        n_suppliers,
        n_consumers,
        suppliers,
        consumers,
        awi.current_step,
        continuous,
        buyos.issues[UNIT_PRICE].min_value,
        sellos.issues[UNIT_PRICE].min_value,
        buyos.issues[UNIT_PRICE].max_value,
        sellos.issues[UNIT_PRICE].max_value,
        buyos.issues[QUANTITY].max_value,
        sellos.issues[QUANTITY].max_value,
        n_prices=n_prices,
    )


def encode_given_offers(
    offers: dict[str, Outcome | None],
    state: OneShotAWI | OneShotState,
    n_suppliers: int,
    n_consumers: int,
    max_group_size: int,
    continuous: bool,
) -> list[tuple[int, int]] | list[tuple[float, float]]:
    encoder = encode_offers_no_time
    normalizer = normalize_offers_no_time
    suppliers = group_partners(state.my_suppliers, n_suppliers, max_group_size)
    consumers = group_partners(state.my_consumers, n_consumers, max_group_size)

    min_iprice = state.current_input_outcome_space.issues[UNIT_PRICE].min_value
    max_iprice = state.current_input_outcome_space.issues[UNIT_PRICE].max_value
    max_iquantity = state.current_input_outcome_space.issues[QUANTITY].max_value
    ioffers = encoder(offers, suppliers, min_iprice, max_iprice)
    if continuous:
        ioffers = normalizer(
            ioffers, min_iprice, max_iprice, 0, max_iquantity, subtract_min_price=False
        )
    min_oprice = state.current_output_outcome_space.issues[UNIT_PRICE].min_value
    max_oprice = state.current_output_outcome_space.issues[UNIT_PRICE].max_value
    max_oquantity = state.current_output_outcome_space.issues[QUANTITY].max_value
    ooffers = encoder(offers, consumers, min_oprice, max_iprice)
    if continuous:
        ooffers = normalizer(
            ooffers, min_oprice, max_oprice, 0, max_oquantity, subtract_min_price=False
        )
    return ioffers + ooffers


def read_offers(
    state: OneShotAWI | OneShotState,
    n_suppliers: int,
    n_consumers: int,
    max_group_size: int,
    continuous: bool,
) -> list[tuple[int, int]] | list[tuple[float, float]]:
    return encode_given_offers(
        offers=state.current_offers,  # type: ignore
        state=state,
        n_suppliers=n_suppliers,
        n_consumers=n_consumers,
        max_group_size=max_group_size,
        continuous=continuous,
    )


def encode_offers_with_time(
    offers: Mapping[str, Outcome | None],
    partner_groups: list[list[str]],
    min_price: int,
    max_price: int,
) -> list[tuple[int, int, int]]:
    """
    Encodes offers from the given partner groups into `n_partners` tuples of quantity, unit-price values.

    Args:
        offers: All received offers. Keys are sources. Sources not in the `partner_groups` will be ignored
        partner_groups: A list of lists of partner IDs each defining a group to be considered together
        min_price: Minimum allowed price
        max_price: Maximum allowed price

    Return:
        A list of quantity, unit-price tuples of length `len(partner_groups)`.
    """
    n_partners = len(partner_groups)
    offer_list: list[tuple[int, int, int]] = [(0, 0, 0) for _ in range(n_partners)]
    for i, partners in enumerate(partner_groups):
        n_read = 0
        curr_offer = dict()
        for partner in partners:
            outcome = offers.get(partner, None)
            if outcome is None:
                continue
            c = curr_offer.get(outcome[TIME], (0, 0))
            curr_offer[outcome[TIME]] = (
                c[0] + outcome[QUANTITY],
                c[1] + outcome[UNIT_PRICE] * outcome[QUANTITY],
            )
            n_read += 1
        if n_read:
            for t, c in curr_offer.items():
                if c[0]:
                    c = (
                        c[0],
                        c[1] / c[0] - min_price,
                    )
                else:
                    c = (0, max_price - min_price)
                curr_offer[t] = c
                offer_list[i]
        else:
            offer_list[i] = (0, 0, 0)
    return offer_list


def encode_offers_no_time(
    offers: Mapping[str, Outcome | None],
    partner_groups: list[list[str]],
    min_price: int,
    max_price: int,
) -> list[tuple[int, int]]:
    """
    Encodes offers from the given partner groups into `n_partners` tuples of quantity, unit-price values.

    Args:
        offers: All received offers. Keys are sources. Sources not in the `partner_groups` will be ignored
        partner_groups: A list of lists of partner IDs each defining a group to be considered together
        min_price: Minimum allowed price
        max_price: Maximum allowed price

    Return:
        A list of quantity, unit-price tuples of length `len(partner_groups)`.
    """
    n_partners = len(partner_groups)
    offer_list: list[tuple[int, int]] = [(0, 0) for _ in range(n_partners)]
    for i, partners in enumerate(partner_groups):
        n_read = 0
        curr_offer = (0, 0)
        for partner in partners:
            outcome = offers.get(partner, None)
            if outcome is None:
                continue
            curr_offer = (
                curr_offer[0] + outcome[QUANTITY],
                curr_offer[1] + outcome[UNIT_PRICE] * outcome[QUANTITY],
            )
            n_read += 1
        if n_read:
            if curr_offer[0]:
                curr_offer = (
                    curr_offer[0],
                    curr_offer[1] / curr_offer[0] - min_price,
                )
            else:
                curr_offer = (0, max_price - min_price)
        offer_list[i] = curr_offer
    return offer_list


def decode_offers_no_time(
    encoded: np.ndarray | list[tuple[int, int]] | list[tuple[float, float]],
    n_suppliers: int,
    n_consumers: int,
    suppliers: list[list[str]],
    consumers: list[list[str]],
    step: int,
    continuous: bool,
    min_buy_price: int,
    min_sell_price: int,
    max_buy_price: int = -1,
    max_sell_price: int = -1,
    max_buy_quantity: int = -1,
    max_sell_quantity: int = -1,
    n_prices: int | None = None,
) -> dict[str, Outcome | None]:
    """
    Inverts `encode_offers_no_time`

    Remarks:
        - max_* are only needed if continuous is True
    """
    n_partners = n_suppliers + n_consumers
    encoded = np.asarray(encoded).flatten()[: n_partners * 2]
    e = np.asarray(encoded).reshape((n_partners, 2))
    encodedl = e.tolist()
    supplier_offers = encodedl[:n_suppliers]
    consumer_offers = encodedl[n_suppliers:]
    if continuous:
        supplier_offers = unnormalize_offers(
            supplier_offers,
            min_buy_price,
            max_buy_price,
            0,
            max_buy_quantity,
            add_min_price=False,
        )
        consumer_offers = unnormalize_offers(
            consumer_offers,
            min_sell_price,
            max_sell_price,
            0,
            max_sell_quantity,
            add_min_price=False,
        )
    responses: dict[str, Outcome | None] = defaultdict(lambda: (0, 0, 0))

    def update_respones(plst, w, is_supplier):
        p = "+".join(plst)
        minprice = min_buy_price if is_supplier else min_sell_price
        maxprice = max_buy_price if is_supplier else max_sell_price
        if w[0] == w[1] == 0:
            responses[p] = None
            return

        price = w[1] + minprice
        if n_prices:
            price *= n_prices / (maxprice - minprice + 1)
        outcome = (int(w[0] + 0.5), step, price)
        r = responses[p]
        if r is None:
            responses[p] = outcome
        else:
            responses[p] = (
                r[0] + outcome[0],
                max(
                    outcome[1], r[1]
                ),  #  we use the largest step here as all steps should be equal anyway
                r[-1] + outcome[-1],
            )

    if len(suppliers) != len(supplier_offers) or len(consumers) != len(consumer_offers):
        raise AssertionError("fdsdf")

    for plst, w in zip(suppliers, supplier_offers, strict=True):
        update_respones(plst, w, True)
    for plst, w in zip(consumers, consumer_offers, strict=True):
        update_respones(plst, w, False)
    result = {
        k: None if v is not None and v[0] == 0 and v[1] == 0 else v
        for k, v in responses.items()
    }
    return result


def normalize_offers_with_time(
    offers: list[tuple[int, int, int]],
    min_price: int,
    max_price: int,
    min_quantity: int,
    max_quantity: int,
) -> list[tuple[float, float, float]]:
    """
    Normalize the offers to values between 0 and 1 for both quantity and unit price
    """
    d = max_price - min_price
    if not d:
        d = 1
    dq = max_quantity - min_quantity
    if not dq:
        dq = 1
    return [
        (float(offer[0] - min_quantity) / dq, offer[1], float(offer[-1]) / d)
        for offer in offers
    ]


def normalize_offers_no_time(
    offers: list[tuple[int, int]],
    min_price: int,
    max_price: int,
    min_quantity: int,
    max_quantity: int,
    subtract_min_price: int = False,
) -> list[tuple[float, float]]:
    """
    Normalize the offers to values between 0 and 1 for both quantity and unit price
    """
    d = max_price - min_price
    if not d:
        d = 1
    dq = max_quantity - min_quantity
    if not dq:
        dq = 1
    if not subtract_min_price:
        min_price = 0
    return [
        (float(offer[0] - min_quantity) / dq, float(offer[1] - min_price) / d)
        for offer in offers
    ]


def unnormalize_offers(
    offers: list[tuple[float, float]],
    min_price: int,
    max_price: int,
    min_quantity: int,
    max_quantity: int,
    add_min_price: bool = False,
) -> list[tuple[int, int]]:
    """
    Reverses `normalize_offers` converting quantities and prices in the range 0,1 to integers
    """
    d = max_price - min_price
    if not d:
        d = 1
    dq = max_quantity - min_quantity
    if not dq:
        dq = 1
    if not add_min_price:
        min_price = 0
    return [
        (int(offer[0] * dq + min_quantity + 0.5), int(offer[1] * d + min_price + 0.5))
        for offer in offers
    ]


def clip_normal(
    x: float,
    mu: float,
    sigma: float,
    n_sigmas: float | int = 3,
    eps: float = 1e-6,
) -> float:
    """
    Normalizes x between 0 and 1 given that it is sampled from a normal (mu, sigma).
    This is actually a very stupid way to do it.
    """
    mn = mu - n_sigmas * sigma
    mx = mu + n_sigmas * sigma
    if abs(mn - mx) < eps:
        return 1.0
    return max(0.0, min(1.0, (x - mn) / (mx - mn)))


T = TypeVar("T", bound=int | float)


def clip(x: T, mn: T = 0, mx: T = 1) -> T:
    return max(mn, min(mx, x))


def discretize_and_clip(x: float, n_bins: int) -> int:
    return min(n_bins - 1, max(0, int(0.5 + (n_bins - 1) * x)))


def normalize_and_clip(x: int, mn: T, mx: T, eps=1e-6) -> float:
    d = mx - mn
    if d < eps:
        return float(mx)
    return clip((x - mn) / d, 0.0, 1.0)
