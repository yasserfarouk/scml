OneShotAWI
==========

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotAWI
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotAWI.accepted_negotiation_requests
      ~OneShotAWI.all_consumers
      ~OneShotAWI.all_suppliers
      ~OneShotAWI.allow_zero_quantity
      ~OneShotAWI.catalog_prices
      ~OneShotAWI.current_balance
      ~OneShotAWI.current_buy_nmis
      ~OneShotAWI.current_buy_offers
      ~OneShotAWI.current_buy_states
      ~OneShotAWI.current_disposal_cost
      ~OneShotAWI.current_exogenous_input_price
      ~OneShotAWI.current_exogenous_input_quantity
      ~OneShotAWI.current_exogenous_output_price
      ~OneShotAWI.current_exogenous_output_quantity
      ~OneShotAWI.current_input_issues
      ~OneShotAWI.current_input_outcome_space
      ~OneShotAWI.current_inventory
      ~OneShotAWI.current_inventory_input
      ~OneShotAWI.current_inventory_output
      ~OneShotAWI.current_negotiation_details
      ~OneShotAWI.current_nmis
      ~OneShotAWI.current_offers
      ~OneShotAWI.current_output_issues
      ~OneShotAWI.current_output_outcome_space
      ~OneShotAWI.current_score
      ~OneShotAWI.current_sell_nmis
      ~OneShotAWI.current_sell_offers
      ~OneShotAWI.current_sell_states
      ~OneShotAWI.current_shortfall_penalty
      ~OneShotAWI.current_states
      ~OneShotAWI.current_step
      ~OneShotAWI.current_storage_cost
      ~OneShotAWI.default_signing_delay
      ~OneShotAWI.exogenous_contract_summary
      ~OneShotAWI.future_sales
      ~OneShotAWI.future_sales_cost
      ~OneShotAWI.future_supplies
      ~OneShotAWI.future_supplies_cost
      ~OneShotAWI.horizon
      ~OneShotAWI.initialized
      ~OneShotAWI.is_exogenous_forced
      ~OneShotAWI.is_first_level
      ~OneShotAWI.is_last_level
      ~OneShotAWI.is_middle_level
      ~OneShotAWI.is_perishable
      ~OneShotAWI.level
      ~OneShotAWI.max_n_lines
      ~OneShotAWI.my_consumers
      ~OneShotAWI.my_input_product
      ~OneShotAWI.my_output_product
      ~OneShotAWI.my_partners
      ~OneShotAWI.my_suppliers
      ~OneShotAWI.n_competitors
      ~OneShotAWI.n_input_negotiations
      ~OneShotAWI.n_lines
      ~OneShotAWI.n_output_negotiations
      ~OneShotAWI.n_processes
      ~OneShotAWI.n_products
      ~OneShotAWI.n_steps
      ~OneShotAWI.needed_sales
      ~OneShotAWI.needed_supplies
      ~OneShotAWI.negotiation_requests
      ~OneShotAWI.params
      ~OneShotAWI.penalties_scale
      ~OneShotAWI.price_multiplier
      ~OneShotAWI.price_range
      ~OneShotAWI.profile
      ~OneShotAWI.quantity_range
      ~OneShotAWI.relative_time
      ~OneShotAWI.requested_negotiations
      ~OneShotAWI.running_mechanism_dicts
      ~OneShotAWI.running_negotiations
      ~OneShotAWI.running_sell_nmis
      ~OneShotAWI.sales
      ~OneShotAWI.sales_cost
      ~OneShotAWI.settings
      ~OneShotAWI.state
      ~OneShotAWI.supplies
      ~OneShotAWI.supplies_cost
      ~OneShotAWI.total_future_sales
      ~OneShotAWI.total_future_supplies
      ~OneShotAWI.total_sales
      ~OneShotAWI.total_supplies
      ~OneShotAWI.trading_prices
      ~OneShotAWI.unsigned_contracts

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotAWI.bb_query
      ~OneShotAWI.bb_read
      ~OneShotAWI.bb_record
      ~OneShotAWI.bb_remove
      ~OneShotAWI.execute
      ~OneShotAWI.is_bankrupt
      ~OneShotAWI.is_system
      ~OneShotAWI.logdebug
      ~OneShotAWI.logdebug_agent
      ~OneShotAWI.logerror
      ~OneShotAWI.logerror_agent
      ~OneShotAWI.loginfo
      ~OneShotAWI.loginfo_agent
      ~OneShotAWI.logwarning
      ~OneShotAWI.logwarning_agent
      ~OneShotAWI.penalty_multiplier
      ~OneShotAWI.reports_at_step
      ~OneShotAWI.reports_of_agent
      ~OneShotAWI.request_negotiation_about
      ~OneShotAWI.run_negotiation
      ~OneShotAWI.run_negotiations
      ~OneShotAWI.total_sales_at
      ~OneShotAWI.total_sales_between
      ~OneShotAWI.total_sales_from
      ~OneShotAWI.total_sales_until
      ~OneShotAWI.total_supplies_at
      ~OneShotAWI.total_supplies_between
      ~OneShotAWI.total_supplies_from
      ~OneShotAWI.total_supplies_until

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: all_consumers
   .. autoattribute:: all_suppliers
   .. autoattribute:: allow_zero_quantity
   .. autoattribute:: catalog_prices
   .. autoattribute:: current_balance
   .. autoattribute:: current_buy_nmis
   .. autoattribute:: current_buy_offers
   .. autoattribute:: current_buy_states
   .. autoattribute:: current_disposal_cost
   .. autoattribute:: current_exogenous_input_price
   .. autoattribute:: current_exogenous_input_quantity
   .. autoattribute:: current_exogenous_output_price
   .. autoattribute:: current_exogenous_output_quantity
   .. autoattribute:: current_input_issues
   .. autoattribute:: current_input_outcome_space
   .. autoattribute:: current_inventory
   .. autoattribute:: current_inventory_input
   .. autoattribute:: current_inventory_output
   .. autoattribute:: current_negotiation_details
   .. autoattribute:: current_nmis
   .. autoattribute:: current_offers
   .. autoattribute:: current_output_issues
   .. autoattribute:: current_output_outcome_space
   .. autoattribute:: current_score
   .. autoattribute:: current_sell_nmis
   .. autoattribute:: current_sell_offers
   .. autoattribute:: current_sell_states
   .. autoattribute:: current_shortfall_penalty
   .. autoattribute:: current_states
   .. autoattribute:: current_step
   .. autoattribute:: current_storage_cost
   .. autoattribute:: default_signing_delay
   .. autoattribute:: exogenous_contract_summary
   .. autoattribute:: future_sales
   .. autoattribute:: future_sales_cost
   .. autoattribute:: future_supplies
   .. autoattribute:: future_supplies_cost
   .. autoattribute:: horizon
   .. autoattribute:: initialized
   .. autoattribute:: is_exogenous_forced
   .. autoattribute:: is_first_level
   .. autoattribute:: is_last_level
   .. autoattribute:: is_middle_level
   .. autoattribute:: is_perishable
   .. autoattribute:: level
   .. autoattribute:: max_n_lines
   .. autoattribute:: my_consumers
   .. autoattribute:: my_input_product
   .. autoattribute:: my_output_product
   .. autoattribute:: my_partners
   .. autoattribute:: my_suppliers
   .. autoattribute:: n_competitors
   .. autoattribute:: n_input_negotiations
   .. autoattribute:: n_lines
   .. autoattribute:: n_output_negotiations
   .. autoattribute:: n_processes
   .. autoattribute:: n_products
   .. autoattribute:: n_steps
   .. autoattribute:: needed_sales
   .. autoattribute:: needed_supplies
   .. autoattribute:: negotiation_requests
   .. autoattribute:: params
   .. autoattribute:: penalties_scale
   .. autoattribute:: price_multiplier
   .. autoattribute:: price_range
   .. autoattribute:: profile
   .. autoattribute:: quantity_range
   .. autoattribute:: relative_time
   .. autoattribute:: requested_negotiations
   .. autoattribute:: running_mechanism_dicts
   .. autoattribute:: running_negotiations
   .. autoattribute:: running_sell_nmis
   .. autoattribute:: sales
   .. autoattribute:: sales_cost
   .. autoattribute:: settings
   .. autoattribute:: state
   .. autoattribute:: supplies
   .. autoattribute:: supplies_cost
   .. autoattribute:: total_future_sales
   .. autoattribute:: total_future_supplies
   .. autoattribute:: total_sales
   .. autoattribute:: total_supplies
   .. autoattribute:: trading_prices
   .. autoattribute:: unsigned_contracts

   .. rubric:: Methods Documentation

   .. automethod:: bb_query
   .. automethod:: bb_read
   .. automethod:: bb_record
   .. automethod:: bb_remove
   .. automethod:: execute
   .. automethod:: is_bankrupt
   .. automethod:: is_system
   .. automethod:: logdebug
   .. automethod:: logdebug_agent
   .. automethod:: logerror
   .. automethod:: logerror_agent
   .. automethod:: loginfo
   .. automethod:: loginfo_agent
   .. automethod:: logwarning
   .. automethod:: logwarning_agent
   .. automethod:: penalty_multiplier
   .. automethod:: reports_at_step
   .. automethod:: reports_of_agent
   .. automethod:: request_negotiation_about
   .. automethod:: run_negotiation
   .. automethod:: run_negotiations
   .. automethod:: total_sales_at
   .. automethod:: total_sales_between
   .. automethod:: total_sales_from
   .. automethod:: total_sales_until
   .. automethod:: total_supplies_at
   .. automethod:: total_supplies_between
   .. automethod:: total_supplies_from
   .. automethod:: total_supplies_until
