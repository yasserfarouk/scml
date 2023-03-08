MarketAwareMovingRangeAgent
===========================

.. currentmodule:: scml.scml2020

.. autoclass:: MarketAwareMovingRangeAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MarketAwareMovingRangeAgent.accepted_negotiation_requests
      ~MarketAwareMovingRangeAgent.awi
      ~MarketAwareMovingRangeAgent.crisp_ufun
      ~MarketAwareMovingRangeAgent.has_cardinal_preferences
      ~MarketAwareMovingRangeAgent.has_preferences
      ~MarketAwareMovingRangeAgent.has_ufun
      ~MarketAwareMovingRangeAgent.id
      ~MarketAwareMovingRangeAgent.initialized
      ~MarketAwareMovingRangeAgent.internal_state
      ~MarketAwareMovingRangeAgent.name
      ~MarketAwareMovingRangeAgent.negotiation_requests
      ~MarketAwareMovingRangeAgent.preferences
      ~MarketAwareMovingRangeAgent.prob_ufun
      ~MarketAwareMovingRangeAgent.requested_negotiations
      ~MarketAwareMovingRangeAgent.reserved_outcome
      ~MarketAwareMovingRangeAgent.reserved_value
      ~MarketAwareMovingRangeAgent.running_negotiations
      ~MarketAwareMovingRangeAgent.short_type_name
      ~MarketAwareMovingRangeAgent.type_name
      ~MarketAwareMovingRangeAgent.type_postfix
      ~MarketAwareMovingRangeAgent.ufun
      ~MarketAwareMovingRangeAgent.unsigned_contracts
      ~MarketAwareMovingRangeAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MarketAwareMovingRangeAgent.before_step
      ~MarketAwareMovingRangeAgent.can_be_produced
      ~MarketAwareMovingRangeAgent.checkpoint
      ~MarketAwareMovingRangeAgent.checkpoint_info
      ~MarketAwareMovingRangeAgent.confirm_production
      ~MarketAwareMovingRangeAgent.create
      ~MarketAwareMovingRangeAgent.create_negotiation_request
      ~MarketAwareMovingRangeAgent.from_checkpoint
      ~MarketAwareMovingRangeAgent.from_config
      ~MarketAwareMovingRangeAgent.init
      ~MarketAwareMovingRangeAgent.init_
      ~MarketAwareMovingRangeAgent.notify
      ~MarketAwareMovingRangeAgent.on_agent_bankrupt
      ~MarketAwareMovingRangeAgent.on_contract_breached
      ~MarketAwareMovingRangeAgent.on_contract_cancelled
      ~MarketAwareMovingRangeAgent.on_contract_cancelled_
      ~MarketAwareMovingRangeAgent.on_contract_executed
      ~MarketAwareMovingRangeAgent.on_contract_signed
      ~MarketAwareMovingRangeAgent.on_contract_signed_
      ~MarketAwareMovingRangeAgent.on_contracts_finalized
      ~MarketAwareMovingRangeAgent.on_event
      ~MarketAwareMovingRangeAgent.on_failures
      ~MarketAwareMovingRangeAgent.on_neg_request_accepted
      ~MarketAwareMovingRangeAgent.on_neg_request_accepted_
      ~MarketAwareMovingRangeAgent.on_neg_request_rejected
      ~MarketAwareMovingRangeAgent.on_neg_request_rejected_
      ~MarketAwareMovingRangeAgent.on_negotiation_failure
      ~MarketAwareMovingRangeAgent.on_negotiation_failure_
      ~MarketAwareMovingRangeAgent.on_negotiation_success
      ~MarketAwareMovingRangeAgent.on_negotiation_success_
      ~MarketAwareMovingRangeAgent.on_preferences_changed
      ~MarketAwareMovingRangeAgent.on_simulation_step_ended
      ~MarketAwareMovingRangeAgent.on_simulation_step_started
      ~MarketAwareMovingRangeAgent.predict_quantity
      ~MarketAwareMovingRangeAgent.read_config
      ~MarketAwareMovingRangeAgent.respond_to_negotiation_request
      ~MarketAwareMovingRangeAgent.respond_to_negotiation_request_
      ~MarketAwareMovingRangeAgent.respond_to_renegotiation_request
      ~MarketAwareMovingRangeAgent.set_preferences
      ~MarketAwareMovingRangeAgent.set_renegotiation_agenda
      ~MarketAwareMovingRangeAgent.sign_all_contracts
      ~MarketAwareMovingRangeAgent.sign_contract
      ~MarketAwareMovingRangeAgent.spawn
      ~MarketAwareMovingRangeAgent.spawn_object
      ~MarketAwareMovingRangeAgent.step
      ~MarketAwareMovingRangeAgent.step_
      ~MarketAwareMovingRangeAgent.to_dict
      ~MarketAwareMovingRangeAgent.trade_prediction_before_step
      ~MarketAwareMovingRangeAgent.trade_prediction_init
      ~MarketAwareMovingRangeAgent.trade_prediction_step

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
   .. autoattribute:: uuid

   .. rubric:: Methods Documentation

   .. automethod:: before_step
   .. automethod:: can_be_produced
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
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
   .. automethod:: predict_quantity
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
   .. automethod:: step
   .. automethod:: step_
   .. automethod:: to_dict
   .. automethod:: trade_prediction_before_step
   .. automethod:: trade_prediction_init
   .. automethod:: trade_prediction_step
