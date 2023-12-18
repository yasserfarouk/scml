"""
Implements the standard version of SCML (starting 2024)
"""


from scml.oneshot.world import OneShotWorld

__all__ = [
    "StdWorld",
    "SCML2024StdWorld",
]


class StdWorld(OneShotWorld):
    """Implements the SCML-OneShot variant of the SCM world.

    Args:
        catalog_prices: An n_products vector (i.e. n_processes+1 vector) giving the catalog price of all products
        profiles: An n_agents list of `OneShotFactoryProfile` objects specifying the private profile of the factory
                  associated with each agent.
        agent_types: An n_agents list of strings/ `OneShotAgent` classes specifying the type of each agent
        agent_params: An n_agents dictionaries giving the parameters of each agent
        catalog_quantities: The quantities in the past for which catalog_prices are the average unit prices. This
                            is used when updating the trading prices. If set to zero then the trading price will
                            follow the market price and will not use the catalog_price (except for products that are
                            never sold in the market for which the trading price will take the default value of the
                            catalog price). If set to a large value (e.g. 10000), the price at which a product is sold
                            will not affect the trading price
        financial_report_period: The number of steps between financial reports. If < 1, it is a fraction of n_steps
        exogenous_force_max: If true, exogenous contracts are forced to be signed independent of the setting of
                             `force_signing`
        compact: If True, no logs will be kept and the whole simulation will use a smaller memory footprint
        n_steps: Number of simulation steps (can be considered as days).
        time_limit: Total time allowed for the complete simulation in seconds.
        neg_n_steps: Number of negotiation steps allowed for all negotiations.
        neg_time_limit: Total time allowed for a complete negotiation in seconds.
        neg_step_time_limit: Total time allowed for a single step of a negotiation. in seconds.
        negotiation_speed: The number of negotiation steps that pass in every simulation step. If 0, negotiations
                           will be guaranteed to finish within a single simulation step
        signing_delay: The number of simulation steps to pass between a contract is concluded and signed
        name: The name of the simulations
        **kwargs: Other parameters that are passed directly to `SCML2020World` constructor.

    """

    pass


class SCML2024StdWorld(StdWorld):
    pass
