MarketAwareIndependentNegotiationsAgent
=======================================

.. currentmodule:: scml.scml2020

.. autoclass:: MarketAwareIndependentNegotiationsAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MarketAwareIndependentNegotiationsAgent.accepted_negotiation_requests
      ~MarketAwareIndependentNegotiationsAgent.awi
      ~MarketAwareIndependentNegotiationsAgent.crisp_ufun
      ~MarketAwareIndependentNegotiationsAgent.has_cardinal_preferences
      ~MarketAwareIndependentNegotiationsAgent.has_preferences
      ~MarketAwareIndependentNegotiationsAgent.has_ufun
      ~MarketAwareIndependentNegotiationsAgent.id
      ~MarketAwareIndependentNegotiationsAgent.initialized
      ~MarketAwareIndependentNegotiationsAgent.internal_state
      ~MarketAwareIndependentNegotiationsAgent.name
      ~MarketAwareIndependentNegotiationsAgent.negotiation_requests
      ~MarketAwareIndependentNegotiationsAgent.preferences
      ~MarketAwareIndependentNegotiationsAgent.prob_ufun
      ~MarketAwareIndependentNegotiationsAgent.requested_negotiations
      ~MarketAwareIndependentNegotiationsAgent.reserved_outcome
      ~MarketAwareIndependentNegotiationsAgent.reserved_value
      ~MarketAwareIndependentNegotiationsAgent.running_negotiations
      ~MarketAwareIndependentNegotiationsAgent.short_type_name
      ~MarketAwareIndependentNegotiationsAgent.type_name
      ~MarketAwareIndependentNegotiationsAgent.type_postfix
      ~MarketAwareIndependentNegotiationsAgent.ufun
      ~MarketAwareIndependentNegotiationsAgent.unsigned_contracts
      ~MarketAwareIndependentNegotiationsAgent.use_trading
      ~MarketAwareIndependentNegotiationsAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MarketAwareIndependentNegotiationsAgent.acceptable_unit_price
      ~MarketAwareIndependentNegotiationsAgent.before_step
      ~MarketAwareIndependentNegotiationsAgent.checkpoint
      ~MarketAwareIndependentNegotiationsAgent.checkpoint_info
      ~MarketAwareIndependentNegotiationsAgent.confirm_production
      ~MarketAwareIndependentNegotiationsAgent.create
      ~MarketAwareIndependentNegotiationsAgent.create_negotiation_request
      ~MarketAwareIndependentNegotiationsAgent.create_ufun
      ~MarketAwareIndependentNegotiationsAgent.from_checkpoint
      ~MarketAwareIndependentNegotiationsAgent.from_config
      ~MarketAwareIndependentNegotiationsAgent.init
      ~MarketAwareIndependentNegotiationsAgent.init_
      ~MarketAwareIndependentNegotiationsAgent.negotiator
      ~MarketAwareIndependentNegotiationsAgent.notify
      ~MarketAwareIndependentNegotiationsAgent.on_agent_bankrupt
      ~MarketAwareIndependentNegotiationsAgent.on_contract_breached
      ~MarketAwareIndependentNegotiationsAgent.on_contract_cancelled
      ~MarketAwareIndependentNegotiationsAgent.on_contract_cancelled_
      ~MarketAwareIndependentNegotiationsAgent.on_contract_executed
      ~MarketAwareIndependentNegotiationsAgent.on_contract_signed
      ~MarketAwareIndependentNegotiationsAgent.on_contract_signed_
      ~MarketAwareIndependentNegotiationsAgent.on_contracts_finalized
      ~MarketAwareIndependentNegotiationsAgent.on_event
      ~MarketAwareIndependentNegotiationsAgent.on_failures
      ~MarketAwareIndependentNegotiationsAgent.on_neg_request_accepted
      ~MarketAwareIndependentNegotiationsAgent.on_neg_request_accepted_
      ~MarketAwareIndependentNegotiationsAgent.on_neg_request_rejected
      ~MarketAwareIndependentNegotiationsAgent.on_neg_request_rejected_
      ~MarketAwareIndependentNegotiationsAgent.on_negotiation_failure
      ~MarketAwareIndependentNegotiationsAgent.on_negotiation_failure_
      ~MarketAwareIndependentNegotiationsAgent.on_negotiation_success
      ~MarketAwareIndependentNegotiationsAgent.on_negotiation_success_
      ~MarketAwareIndependentNegotiationsAgent.on_preferences_changed
      ~MarketAwareIndependentNegotiationsAgent.on_simulation_step_ended
      ~MarketAwareIndependentNegotiationsAgent.on_simulation_step_started
      ~MarketAwareIndependentNegotiationsAgent.read_config
      ~MarketAwareIndependentNegotiationsAgent.respond_to_negotiation_request
      ~MarketAwareIndependentNegotiationsAgent.respond_to_negotiation_request_
      ~MarketAwareIndependentNegotiationsAgent.respond_to_renegotiation_request
      ~MarketAwareIndependentNegotiationsAgent.set_preferences
      ~MarketAwareIndependentNegotiationsAgent.set_renegotiation_agenda
      ~MarketAwareIndependentNegotiationsAgent.sign_all_contracts
      ~MarketAwareIndependentNegotiationsAgent.sign_contract
      ~MarketAwareIndependentNegotiationsAgent.spawn
      ~MarketAwareIndependentNegotiationsAgent.spawn_object
      ~MarketAwareIndependentNegotiationsAgent.start_negotiations
      ~MarketAwareIndependentNegotiationsAgent.step
      ~MarketAwareIndependentNegotiationsAgent.step_
      ~MarketAwareIndependentNegotiationsAgent.target_quantities
      ~MarketAwareIndependentNegotiationsAgent.target_quantity
      ~MarketAwareIndependentNegotiationsAgent.to_dict
      ~MarketAwareIndependentNegotiationsAgent.trade_prediction_before_step
      ~MarketAwareIndependentNegotiationsAgent.trade_prediction_init
      ~MarketAwareIndependentNegotiationsAgent.trade_prediction_step

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: has_ufun
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
