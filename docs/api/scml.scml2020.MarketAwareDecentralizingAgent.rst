MarketAwareDecentralizingAgent
==============================

.. currentmodule:: scml.scml2020

.. autoclass:: MarketAwareDecentralizingAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MarketAwareDecentralizingAgent.accepted_negotiation_requests
      ~MarketAwareDecentralizingAgent.awi
      ~MarketAwareDecentralizingAgent.crisp_ufun
      ~MarketAwareDecentralizingAgent.has_cardinal_preferences
      ~MarketAwareDecentralizingAgent.has_preferences
      ~MarketAwareDecentralizingAgent.id
      ~MarketAwareDecentralizingAgent.initialized
      ~MarketAwareDecentralizingAgent.internal_state
      ~MarketAwareDecentralizingAgent.name
      ~MarketAwareDecentralizingAgent.negotiation_requests
      ~MarketAwareDecentralizingAgent.preferences
      ~MarketAwareDecentralizingAgent.prob_ufun
      ~MarketAwareDecentralizingAgent.requested_negotiations
      ~MarketAwareDecentralizingAgent.reserved_outcome
      ~MarketAwareDecentralizingAgent.reserved_value
      ~MarketAwareDecentralizingAgent.running_negotiations
      ~MarketAwareDecentralizingAgent.short_type_name
      ~MarketAwareDecentralizingAgent.type_name
      ~MarketAwareDecentralizingAgent.type_postfix
      ~MarketAwareDecentralizingAgent.ufun
      ~MarketAwareDecentralizingAgent.unsigned_contracts
      ~MarketAwareDecentralizingAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MarketAwareDecentralizingAgent.acceptable_unit_price
      ~MarketAwareDecentralizingAgent.before_step
      ~MarketAwareDecentralizingAgent.can_be_produced
      ~MarketAwareDecentralizingAgent.checkpoint
      ~MarketAwareDecentralizingAgent.checkpoint_info
      ~MarketAwareDecentralizingAgent.confirm_production
      ~MarketAwareDecentralizingAgent.create
      ~MarketAwareDecentralizingAgent.create_negotiation_request
      ~MarketAwareDecentralizingAgent.from_checkpoint
      ~MarketAwareDecentralizingAgent.from_config
      ~MarketAwareDecentralizingAgent.init
      ~MarketAwareDecentralizingAgent.init_
      ~MarketAwareDecentralizingAgent.notify
      ~MarketAwareDecentralizingAgent.on_agent_bankrupt
      ~MarketAwareDecentralizingAgent.on_contract_breached
      ~MarketAwareDecentralizingAgent.on_contract_cancelled
      ~MarketAwareDecentralizingAgent.on_contract_cancelled_
      ~MarketAwareDecentralizingAgent.on_contract_executed
      ~MarketAwareDecentralizingAgent.on_contract_signed
      ~MarketAwareDecentralizingAgent.on_contract_signed_
      ~MarketAwareDecentralizingAgent.on_contracts_finalized
      ~MarketAwareDecentralizingAgent.on_event
      ~MarketAwareDecentralizingAgent.on_failures
      ~MarketAwareDecentralizingAgent.on_neg_request_accepted
      ~MarketAwareDecentralizingAgent.on_neg_request_accepted_
      ~MarketAwareDecentralizingAgent.on_neg_request_rejected
      ~MarketAwareDecentralizingAgent.on_neg_request_rejected_
      ~MarketAwareDecentralizingAgent.on_negotiation_failure
      ~MarketAwareDecentralizingAgent.on_negotiation_failure_
      ~MarketAwareDecentralizingAgent.on_negotiation_success
      ~MarketAwareDecentralizingAgent.on_negotiation_success_
      ~MarketAwareDecentralizingAgent.on_preferences_changed
      ~MarketAwareDecentralizingAgent.on_simulation_step_ended
      ~MarketAwareDecentralizingAgent.on_simulation_step_started
      ~MarketAwareDecentralizingAgent.predict_quantity
      ~MarketAwareDecentralizingAgent.read_config
      ~MarketAwareDecentralizingAgent.respond_to_negotiation_request
      ~MarketAwareDecentralizingAgent.respond_to_negotiation_request_
      ~MarketAwareDecentralizingAgent.respond_to_renegotiation_request
      ~MarketAwareDecentralizingAgent.set_preferences
      ~MarketAwareDecentralizingAgent.set_renegotiation_agenda
      ~MarketAwareDecentralizingAgent.sign_all_contracts
      ~MarketAwareDecentralizingAgent.sign_contract
      ~MarketAwareDecentralizingAgent.spawn
      ~MarketAwareDecentralizingAgent.spawn_object
      ~MarketAwareDecentralizingAgent.step
      ~MarketAwareDecentralizingAgent.step_
      ~MarketAwareDecentralizingAgent.target_quantities
      ~MarketAwareDecentralizingAgent.target_quantity
      ~MarketAwareDecentralizingAgent.to_dict
      ~MarketAwareDecentralizingAgent.trade_prediction_before_step
      ~MarketAwareDecentralizingAgent.trade_prediction_init
      ~MarketAwareDecentralizingAgent.trade_prediction_step

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
   .. autoattribute:: uuid

   .. rubric:: Methods Documentation

   .. automethod:: acceptable_unit_price
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
   .. automethod:: target_quantities
   .. automethod:: target_quantity
   .. automethod:: to_dict
   .. automethod:: trade_prediction_before_step
   .. automethod:: trade_prediction_init
   .. automethod:: trade_prediction_step
