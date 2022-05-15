MarketAwareBuyCheapSellExpensiveAgent
=====================================

.. currentmodule:: scml.scml2020

.. autoclass:: MarketAwareBuyCheapSellExpensiveAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MarketAwareBuyCheapSellExpensiveAgent.accepted_negotiation_requests
      ~MarketAwareBuyCheapSellExpensiveAgent.awi
      ~MarketAwareBuyCheapSellExpensiveAgent.crisp_ufun
      ~MarketAwareBuyCheapSellExpensiveAgent.has_cardinal_preferences
      ~MarketAwareBuyCheapSellExpensiveAgent.has_preferences
      ~MarketAwareBuyCheapSellExpensiveAgent.id
      ~MarketAwareBuyCheapSellExpensiveAgent.initialized
      ~MarketAwareBuyCheapSellExpensiveAgent.internal_state
      ~MarketAwareBuyCheapSellExpensiveAgent.name
      ~MarketAwareBuyCheapSellExpensiveAgent.negotiation_requests
      ~MarketAwareBuyCheapSellExpensiveAgent.preferences
      ~MarketAwareBuyCheapSellExpensiveAgent.prob_ufun
      ~MarketAwareBuyCheapSellExpensiveAgent.requested_negotiations
      ~MarketAwareBuyCheapSellExpensiveAgent.reserved_outcome
      ~MarketAwareBuyCheapSellExpensiveAgent.reserved_value
      ~MarketAwareBuyCheapSellExpensiveAgent.running_negotiations
      ~MarketAwareBuyCheapSellExpensiveAgent.short_type_name
      ~MarketAwareBuyCheapSellExpensiveAgent.type_name
      ~MarketAwareBuyCheapSellExpensiveAgent.type_postfix
      ~MarketAwareBuyCheapSellExpensiveAgent.ufun
      ~MarketAwareBuyCheapSellExpensiveAgent.unsigned_contracts
      ~MarketAwareBuyCheapSellExpensiveAgent.use_trading
      ~MarketAwareBuyCheapSellExpensiveAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MarketAwareBuyCheapSellExpensiveAgent.acceptable_unit_price
      ~MarketAwareBuyCheapSellExpensiveAgent.before_step
      ~MarketAwareBuyCheapSellExpensiveAgent.checkpoint
      ~MarketAwareBuyCheapSellExpensiveAgent.checkpoint_info
      ~MarketAwareBuyCheapSellExpensiveAgent.confirm_production
      ~MarketAwareBuyCheapSellExpensiveAgent.create
      ~MarketAwareBuyCheapSellExpensiveAgent.create_negotiation_request
      ~MarketAwareBuyCheapSellExpensiveAgent.create_ufun
      ~MarketAwareBuyCheapSellExpensiveAgent.from_checkpoint
      ~MarketAwareBuyCheapSellExpensiveAgent.from_config
      ~MarketAwareBuyCheapSellExpensiveAgent.init
      ~MarketAwareBuyCheapSellExpensiveAgent.init_
      ~MarketAwareBuyCheapSellExpensiveAgent.negotiator
      ~MarketAwareBuyCheapSellExpensiveAgent.notify
      ~MarketAwareBuyCheapSellExpensiveAgent.on_agent_bankrupt
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contract_breached
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contract_cancelled
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contract_cancelled_
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contract_executed
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contract_signed
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contract_signed_
      ~MarketAwareBuyCheapSellExpensiveAgent.on_contracts_finalized
      ~MarketAwareBuyCheapSellExpensiveAgent.on_event
      ~MarketAwareBuyCheapSellExpensiveAgent.on_failures
      ~MarketAwareBuyCheapSellExpensiveAgent.on_neg_request_accepted
      ~MarketAwareBuyCheapSellExpensiveAgent.on_neg_request_accepted_
      ~MarketAwareBuyCheapSellExpensiveAgent.on_neg_request_rejected
      ~MarketAwareBuyCheapSellExpensiveAgent.on_neg_request_rejected_
      ~MarketAwareBuyCheapSellExpensiveAgent.on_negotiation_failure
      ~MarketAwareBuyCheapSellExpensiveAgent.on_negotiation_failure_
      ~MarketAwareBuyCheapSellExpensiveAgent.on_negotiation_success
      ~MarketAwareBuyCheapSellExpensiveAgent.on_negotiation_success_
      ~MarketAwareBuyCheapSellExpensiveAgent.on_preferences_changed
      ~MarketAwareBuyCheapSellExpensiveAgent.on_simulation_step_ended
      ~MarketAwareBuyCheapSellExpensiveAgent.on_simulation_step_started
      ~MarketAwareBuyCheapSellExpensiveAgent.read_config
      ~MarketAwareBuyCheapSellExpensiveAgent.respond_to_negotiation_request
      ~MarketAwareBuyCheapSellExpensiveAgent.respond_to_negotiation_request_
      ~MarketAwareBuyCheapSellExpensiveAgent.respond_to_renegotiation_request
      ~MarketAwareBuyCheapSellExpensiveAgent.set_preferences
      ~MarketAwareBuyCheapSellExpensiveAgent.set_renegotiation_agenda
      ~MarketAwareBuyCheapSellExpensiveAgent.sign_all_contracts
      ~MarketAwareBuyCheapSellExpensiveAgent.sign_contract
      ~MarketAwareBuyCheapSellExpensiveAgent.spawn
      ~MarketAwareBuyCheapSellExpensiveAgent.spawn_object
      ~MarketAwareBuyCheapSellExpensiveAgent.start_negotiations
      ~MarketAwareBuyCheapSellExpensiveAgent.step
      ~MarketAwareBuyCheapSellExpensiveAgent.step_
      ~MarketAwareBuyCheapSellExpensiveAgent.target_quantities
      ~MarketAwareBuyCheapSellExpensiveAgent.target_quantity
      ~MarketAwareBuyCheapSellExpensiveAgent.to_dict
      ~MarketAwareBuyCheapSellExpensiveAgent.trade_prediction_before_step
      ~MarketAwareBuyCheapSellExpensiveAgent.trade_prediction_init
      ~MarketAwareBuyCheapSellExpensiveAgent.trade_prediction_step

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: id
   .. autoattribute:: initialized
   .. autoattribute:: internal_state
   .. autoattribute:: name
   .. autoattribute:: negotiation_requests
   .. autoattribute:: preferences
   .. autoattribute:: prob_ufun
   .. autoattribute:: requested_negotiations
   .. autoattribute:: reserved_outcome
   .. autoattribute:: reserved_value
   .. autoattribute:: running_negotiations
   .. autoattribute:: short_type_name
   .. autoattribute:: type_name
   .. autoattribute:: type_postfix
   .. autoattribute:: ufun
   .. autoattribute:: unsigned_contracts
   .. autoattribute:: use_trading
   .. autoattribute:: uuid

   .. rubric:: Methods Documentation

   .. automethod:: acceptable_unit_price
   .. automethod:: before_step
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: create_ufun
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: negotiator
   .. automethod:: notify
   .. automethod:: on_agent_bankrupt
   .. automethod:: on_contract_breached
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_cancelled_
   .. automethod:: on_contract_executed
   .. automethod:: on_contract_signed
   .. automethod:: on_contract_signed_
   .. automethod:: on_contracts_finalized
   .. automethod:: on_event
   .. automethod:: on_failures
   .. automethod:: on_neg_request_accepted
   .. automethod:: on_neg_request_accepted_
   .. automethod:: on_neg_request_rejected
   .. automethod:: on_neg_request_rejected_
   .. automethod:: on_negotiation_failure
   .. automethod:: on_negotiation_failure_
   .. automethod:: on_negotiation_success
   .. automethod:: on_negotiation_success_
   .. automethod:: on_preferences_changed
   .. automethod:: on_simulation_step_ended
   .. automethod:: on_simulation_step_started
   .. automethod:: read_config
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: set_preferences
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: start_negotiations
   .. automethod:: step
   .. automethod:: step_
   .. automethod:: target_quantities
   .. automethod:: target_quantity
   .. automethod:: to_dict
   .. automethod:: trade_prediction_before_step
   .. automethod:: trade_prediction_init
   .. automethod:: trade_prediction_step
