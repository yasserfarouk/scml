from typing import Any

from scml.oneshot.world import DUMMY_AGENT_BEGINNING, SCMLBaseWorld

__all__ = [
    "StdWorld",
    "SCML2024StdWorld",
    "STD_DEFAULT_PARAMS",
    "STD_DEFAULT_PARAMS2024",
    "DUMMY_AGENT_BEGINNING",
]

STD_DEFAULT_PARAMS = dict(
    perishable=False,
    horizon=10,
    n_processes=2,
    disposal_cost=0,
    disposal_cost_dev=0,
    storage_cost=(0.01, 0.05),
    storage_cost_dev=0,
    price_range_fraction=0.1,
    price_multiplier=0.0,
    wide_price_range=False,
    one_time_per_negotiation=False,
    quantity_multiplier=3,
)
STD_DEFAULT_PARAMS2024 = STD_DEFAULT_PARAMS


class StdWorld(SCMLBaseWorld):
    """The world representing the base standard simulation (starting SCML 2024)"""

    def __init__(
        self,
        *args,
        horizon=STD_DEFAULT_PARAMS["horizon"],
        price_range_fraction=STD_DEFAULT_PARAMS["price_range_fraction"],
        price_multiplier=STD_DEFAULT_PARAMS["price_multiplier"],
        wide_price_range=STD_DEFAULT_PARAMS["wide_price_range"],
        one_time_per_negotiation=STD_DEFAULT_PARAMS["one_time_per_negotiation"],
        perishable=STD_DEFAULT_PARAMS["perishable"],
        quantity_multiplier=STD_DEFAULT_PARAMS["quantity_multiplier"],
        **kwargs
    ):
        super().__init__(
            *args,
            price_range_fraction=price_range_fraction,  # type: ignore
            horizon=horizon,  # type: ignore
            one_time_per_negotiation=one_time_per_negotiation,  # type: ignore
            perishable=perishable,  # type: ignore
            quantity_multiplier=quantity_multiplier,  # type: ignore
            price_multiplier=price_multiplier,  # type: ignore
            wide_price_range=wide_price_range,  # type: ignore
            **kwargs
        )

    @classmethod
    def generate(
        cls,
        *args,
        n_processes=STD_DEFAULT_PARAMS["n_processes"],
        disposal_cost=STD_DEFAULT_PARAMS["disposal_cost"],
        disposal_cost_dev=STD_DEFAULT_PARAMS["disposal_cost_dev"],
        storage_cost=STD_DEFAULT_PARAMS["storage_cost"],
        storage_cost_dev=STD_DEFAULT_PARAMS["storage_cost_dev"],
        perishable=STD_DEFAULT_PARAMS["perishable"],
        **kwargs
    ) -> dict[str, Any]:
        """
        Generates the configuration for a world

        Remarks:
            - This method just sets the defaults differently to create a std instead of a oneshot world.

        """
        return super().generate(
            *args,
            n_processes=n_processes,  # type: ignore
            disposal_cost=disposal_cost,
            disposal_cost_dev=disposal_cost_dev,
            storage_cost=storage_cost,
            storage_cost_dev=storage_cost_dev,
            perishable=perishable,  # type: ignore
            **kwargs
        )


class SCML2024StdWorld(StdWorld):
    """The SCML-standard simulation as used in [SCML 2024](https://scml.cs.brown.edu)"""

    pass
