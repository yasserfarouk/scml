from scml.oneshot.awi import OneShotAWI

__all__ = ["StdAWI"]


class StdAWI(OneShotAWI):
    """
    The agent world interface for the one-shot game.

    This class contains all the methods needed to access the simulation to
    extract information which are divided into 4 groups:

    Static World Information:
        Information about the world and the agent that does not change over
        time. These include:

        A. Market Information:
          - **n_products**: Number of products in the production chain.
          - **n_processes**: Number of processes in the production chain.
          - **n_competitors**: Number of other factories on the same production level.
          - **all_suppliers**: A list of all suppliers by product.
          - **all_consumers**: A list of all consumers by product.
          - **proudction_capacities**: The total production capacity (i.e. number of lines)
                                     for each production level (i.e. manufacturing process).
          - **is_system**: Is the given system ID corresponding to a system agent?
          - **is_bankrupt**: Is the given agent bankrupt? None asks about self
          - **catalog_prices**: A list of the catalog prices (by product).
          - **price_multiplier**: The multiplier multiplied by the trading/catalog price
            when the negotiation agendas are created to decide the maximum and lower quantities.
          - **is_exogenous_forced**: Are exogenous contracts always forced or can the
            agent decide not to sign them.
          - **current_step**: Current simulation step (inherited from `negmas.situated.AgentWorldInterface` ).
          - **n_steps**: Number of simulation steps (inherited from `negmas.situated.AgentWorldInterface` ).
          - **relative_time**: fraction of the simulation completed (inherited from `negmas.situated.AgentWorldInterface`).
          - **state**: The full state of the agent ( `OneShotState` ).
          - **settings* The system settings (inherited from `negmas.situated.AgentWorldInterface` ).
          - **quantity_range* The maximum quantity in all negotiation agendas (new in 0.6.1)
          - **price_range* The maximum number of different prices in any negotiation agenda (new in 0.6.1)
          - **horizon* The negotiation horizon for delivery dates. A value greater than zero indicates that you can get
            agreements about future deliveries.

        B. Agent Information:
          - **profile**: Gives the agent profile including its production cost, number
            of production lines, input product index, mean of its delivery
            penalties, mean of its disposal costs, standard deviation of its
            shortfall penalties and standard deviation of its disposal costs.
            See `OneShotProfile` for full description. This information is private
            information and no other agent knows it.
          - **n_lines**: the number of production lines in the factory (private information).
          - **is_first_level**: Is the agent in the first production level (i.e. it is an
            input agent that buys the raw material).
          - **is_last_level**: Is the agent in the last production level (i.e. it is an
            output agent that sells the final product).
          - **is_middle_level**: Is the agent neither a first level nor a last level agent
          - **my_input_product**: The input product to the factory controlled by the agent.
          - **my_output_product**: The output product from the factory controlled by the agent.
          - **level**: The production level which is numerically the same as the input product.
          - **my_suppliers**: A list of IDs for all suppliers to the agent (i.e. agents
            that can sell the input product of the agent).
          - **my_consumers**: A list of IDs for all consumers to the agent (i.e. agents
            that can buy the output product of the agent).
          - **penalties_scale**: The scale at which to calculate disposal cost/delivery
            penalties. "trading" and "catalog" mean trading and
            catalog prices. "unit" means the contract's unit price
            while "none" means that disposal cost/shortfall penalty
            are absolute.
          - **n_input_negotiations**: Number of negotiations with suppliers.
          - **n_output_negotiations**: Number of negotiations with consumers.

    Dynamic World Information:
        Information about the world and the agent that changes over time.

        A. Market Information:
          - **trading_prices**: The trading prices of all products. This information
            is only available if `publish_trading_prices` is
            set in the world.
          - **exogenous_contract_summary**: A list of n_products tuples each giving
            the total quantity and average price of
            exogenous contracts for a product. This
            information is only available if
            `publish_exogenous_summary` is set in
            the world.
          - **is_perishable**: Are all products perishable?

        B. Other Agents' Information:
          - **reports_of_agent**: Gives all past financial reports of a given agent.
            See `FinancialReport` for details.
          - **reports_at_step**: Gives all reports of all agents at a given step.
            See `FinancialReport` for details.

        C. Current Negotiations Information:
          - **current_input_outcome_space**: The current outcome-space for all negotiations to buy
            the input product of the agent. If the agent is at level zero, this will have no issues.
          - **current_output_outcome_space**: The current outcome-space for all negotiations to buy
            the output product of the agent. If the agent
            is at level n_products - 1, this will have no issues.
          - **current_negotiation_details**: Details on all current negotiations separated into "buy"
            and "sell" dictionaries.

          Useful helpers about current negotiations:

          - **current_input_issues**: The current issues for all negotiations to buy
            the input product of the agent. If the agent
            is at level zero, this will be empty.
            This is exactly the same as current_input_outcome_space.issues
          - **current_output_issues**: The current issues for all negotiations to buy
            the output product of the agent. If the agent
            is at level n_products - 1, this will be empty.
            This is exactly the same as current_output_outcome_space.issues
          - **current_buy_nmis**: All NMIs for current buy negotiations.
          - **current_sell_nmis**: All NMIs for current sell negotiations.
          - **current_nmis**: All states for current negotiations.
          - **current_buy_states**: All states for current buy negotiations.
          - **current_sell_states**: All states for current sell negotiations.
          - **current_states**: All states for current negotiations.
          - **current_buy_offers**: All offers for current buy negotiations.
          - **current_sell_offers**: All offers for current sell negotiations.
          - **current_offers**: All offers for current negotiations.
          - **running_buy_nmis**: All NMIs for running buy negotiations.
          - **running_sell_nmis**: All NMIs for running sell negotiations.
          - **running_nmis**: All states for running negotiations.
          - **running_buy_states**: All states for running buy negotiations.
          - **running_sell_states**: All states for running sell negotiations.
          - **running_states**: All states for running negotiations.

        D. Agent Information:
          - **current_exogenous_input_quantity**: The total quantity the agent have
            in its input exogenous contract.
          - **current_exogenous_input_price**: The total price of the agent's
            input exogenous contract.
          - **current_exogenous_output_quantity**: The total quantity the agent have
            in its output exogenous contract.
          - **current_exogenous_output_price**: The total price of the agent's
            output exogenous contract
          - **current_disposal_cost**: The disposal cost per unit item in the current
            step.
          - **current_shortfall_penalty**: The shortfall penalty per unit item in the current
            step.
          - **current_balance**: The current balance of the agent
          - **current_score**: The current score (balance / initial balance) of the agent
          - **current_inventory_input**: The total quantity remaining in the inventory of the input product
          - **current_inventory_output**: The total quantity remaining in the inventory of the output product
          - **current_inventory**: The total quantity remaining in the inventory of the input and output product

        E. Sales and Supplies (quantities) for today:
          - **sales**: Today's sales per customer so far.
          - **supplies**: Today's supplies per supplier so far.
          - **total_sales**: Today's total sales so far.
          - **total_supplies**: Today's total supplies so far.
          - **needed_sales**: Today's needed sales as of now (exogenous input + total supplies - exogenous output - total sales so far).
          - **needed_supplies**: Today's needed supplies  as of now (exogenous output + total sales - exogenous input - total supplies so far).
          - **future_sales**: Future quantity of the output product in standing contracts not executed nor nullified.
          - **future_supplies**: Future quantity of the input product in standing contracts not executed nor nullified.
          - **total_future_sales**: Total future quantity of the output product in standing contracts not executed nor nullified.
          - **total_future_supplies**: Total future quantity of the input product in standing contracts not executed nor nullified.
          - **total_future_sales_between**: Total future sale quantities between the given two simulated days (non-exogenous).
          - **total_future_supplies_between**: Total future supply quantities between the given two simulated days (non-exogenous).
          - **total_future_sales_until**: Total future sale quantities between tomorrow and the given day (non-exogenous).
          - **total_future_supplies_until**: Total future supply quantities between tomorrow and the given day (non-exogenous).
          - **total_future_sales_at**: Total future sale quantities at the given day (non-exogenous).
          - **total_future_supplies_at**: Total future supply quantities at the given day (non-exogenous).
          - **future_sales_cost**: Future total_cost of the output product in standing contracts not executed nor nullified.
          - **future_supplies_cost**: Future total cost of the input product in standing contracts not executed nor nullified.


    Services (All inherited from `negmas.situated.AgentWorldInterface`):
      - **logdebug/loginfo/logwarning/logerror**: Logs to the world log at the given log level.
      - **logdebug_agent/loginf_agnet/...**: Logs to the agent specific log at the given log level.
      - **bb_query**: Queries the bulletin-board.
      - **bb_read**: Read a section of the bulletin-board.

    """
