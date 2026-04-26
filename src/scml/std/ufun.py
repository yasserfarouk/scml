from scml.oneshot.ufun import OneShotUFun, UFunLimit, UtilityInfo

__all__ = ["StdUFun", "UFunLimit", "UtilityInfo"]


class StdUFun(OneShotUFun):
    """
    Calculates the utility function of a list of contracts or offers.

    Args:
        force_exogenous: Is the agent forced to accept exogenous contracts
                         given through `ex_*` arguments?
        ex_pin: total price of exogenous inputs for this agent
        ex_qin: total quantity of exogenous inputs for this agent
        ex_pout: total price of exogenous outputs for this agent
        ex_qout: total quantity of exogenous outputs for this agent.
        cost: production cost of the agent.
        disposal_cost: disposal cost per unit of input/output.
        shortfall_penalty: penalty for failure to deliver one unit of output.
        input_agent: Is the agent an input agent which means that its input
                     product is the raw material
        output_agent: Is the agent an output agent which means that its output
                      product is the final product
        n_lines: Number of production lines. If None, will be read through the AWI.
        input_product: Index of the input product. If None, will be read through
                       the AWI
        input_qrange: A 2-int tuple giving the range of input quantities negotiated.
                      If not given will be read through the AWI
        input_prange: A 2-int tuple giving the range of input unit prices negotiated.
                      If not given will be read through the AWI
        output_qrange: A 2-int tuple giving the range of output quantities negotiated.
                      If not given will be read through the AWI
        output_prange: A 2-int tuple giving the range of output unit prices negotiated.
                      If not given will be read through the AWI
        n_input_negs: How many input negotiations are allowed. If not given, it
                      will be the number of suppliers as given by the AWI
        n_output_negs: How many output negotiations are allowed. If not given, it
                      will be the number of consumers as given by the AWI
        current_step: Current simulation step. Needed only for `ufun_range`
                      when returning best outcomes
        normalized: If given the values returned by `from_*`, `utility_range`
                    and `__call__` will all be normalized between zero and one.

    Remarks:
        - The utility function assumes that the agent will have to pay for
          all its input products but will receive money only for the output
          products it could generate and sell.
        - The utility function respects production capacity (n. lines). The
          agent cannot produce more than the number of lines it has.
        - disposal cost is paid for items bought but not produced only. Items
          consumed in production (i.e. sold) are not counted.
    """
