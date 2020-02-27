from scml.scml2020.services import FactorySimulator

__all__ = ["Simulation"]


class Simulation:
    """
    Provides a simulator to the agent.

    Provides:
        - `simulator` (FactorySimulator):  A simulator that can be used to simulate the effect of contracts on the
          future of the factory

    Hooks Into:
        - `init`
        - `step`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator: FactorySimulator = None

    def init(self):
        self.simulator = FactorySimulator(
            profile=self.awi.profile,
            initial_balance=self.awi.state.balance,
            bankruptcy_limit=self.awi.settings["bankruptcy_limit"],
            spot_market_global_loss=self.awi.settings["spot_market_global_loss"],
            catalog_prices=self.awi.catalog_prices,
            n_steps=self.awi.n_steps,
            initial_inventory=self.awi.state.inventory,
        )
        super().init()

    def step(self):
        self.simulator.set_state(
            self.awi.current_step,
            self.awi.state.inventory,
            self.awi.state.balance,
            self.awi.state.commands,
        )
        self.simulator.fix_before(self.awi.current_step)
        super().step()
