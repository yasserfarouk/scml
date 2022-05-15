MarketAwareIndDecentralizingAgent
=================================

.. currentmodule:: scml.scml2020

.. autoclass:: MarketAwareIndDecentralizingAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MarketAwareIndDecentralizingAgent.accepted_negotiation_requests
      ~MarketAwareIndDecentralizingAgent.awi
      ~MarketAwareIndDecentralizingAgent.crisp_ufun
      ~MarketAwareIndDecentralizingAgent.has_cardinal_preferences
      ~MarketAwareIndDecentralizingAgent.has_preferences
      ~MarketAwareIndDecentralizingAgent.id
      ~MarketAwareIndDecentralizingAgent.initialized
      ~MarketAwareIndDecentralizingAgent.internal_state
      ~MarketAwareIndDecentralizingAgent.name
      ~MarketAwareIndDecentralizingAgent.negotiation_requests
      ~MarketAwareIndDecentralizingAgent.preferences
      ~MarketAwareIndDecentralizingAgent.prob_ufun
      ~MarketAwareIndDecentralizingAgent.requested_negotiations
      ~MarketAwareIndDecentralizingAgent.reserved_outcome
      ~MarketAwareIndDecentralizingAgent.reserved_value
      ~MarketAwareIndDecentralizingAgent.running_negotiations
      ~MarketAwareIndDecentralizingAgent.short_type_name
      ~MarketAwareIndDecentralizingAgent.type_name
      ~MarketAwareIndDecentralizingAgent.type_postfix
      ~MarketAwareIndDecentralizingAgent.ufun
      ~MarketAwareIndDecentralizingAgent.unsigned_contracts
      ~MarketAwareIndDecentralizingAgent.use_trading
      ~MarketAwareIndDecentralizingAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MarketAwareIndDecentralizingAgent.acceptable_unit_price
      ~MarketAwareIndDecentralizingAgent.before_step
      ~MarketAwareIndDecentralizingAgent.can_be_produced
      ~MarketAwareIndDecentralizingAgent.checkpoint
      ~MarketAwareIndDecentralizingAgent.checkpoint_info
      ~MarketAwareIndDecentralizingAgent.confirm_production
      ~MarketAwareIndDecentralizingAgent.create
      ~MarketAwareIndDecentralizingAgent.create_negotiation_request
      ~MarketAwareIndDecentralizingAgent.create_ufun
      ~MarketAwareIndDecentralizingAgent.from_checkpoint
      ~MarketAwareIndDecentralizingAgent.from_config
      ~MarketAwareIndDecentralizingAgent.init
      ~MarketAwareIndDecentralizingAgent.init_
      ~MarketAwareIndDecentralizingAgent.negotiator
      ~MarketAwareIndDecentralizingAgent.notify
      ~MarketAwareIndDecentralizingAgent.on_agent_bankrupt
      ~MarketAwareIndDecentralizingAgent.on_contract_breached
      ~MarketAwareIndDecentralizingAgent.on_contract_cancelled
      ~MarketAwareIndDecentralizingAgent.on_contract_cancelled_
      ~MarketAwareIndDecentralizingAgent.on_contract_executed
      ~MarketAwareIndDecentralizingAgent.on_contract_signed
      ~MarketAwareIndDecentralizingAgent.on_contract_signed_
      ~MarketAwareIndDecentralizingAgent.on_contracts_finalized
      ~MarketAwareIndDecentralizingAgent.on_event
      ~MarketAwareIndDecentralizingAgent.on_failures
      ~MarketAwareIndDecentralizingAgent.on_neg_request_accepted
      ~MarketAwareIndDecentralizingAgent.on_neg_request_accepted_
      ~MarketAwareIndDecentralizingAgent.on_neg_request_rejected
      ~MarketAwareIndDecentralizingAgent.on_neg_request_rejected_
      ~MarketAwareIndDecentralizingAgent.on_negotiation_failure
      ~MarketAwareIndDecentralizingAgent.on_negotiation_failure_
      ~MarketAwareIndDecentralizingAgent.on_negotiation_success
      ~MarketAwareIndDecentralizingAgent.on_negotiation_success_
      ~MarketAwareIndDecentralizingAgent.on_preferences_changed
      ~MarketAwareIndDecentralizingAgent.on_simulation_step_ended
      ~MarketAwareIndDecentralizingAgent.on_simulation_step_started
      ~MarketAwareIndDecentralizingAgent.predict_quantity
      ~MarketAwareIndDecentralizingAgent.read_config
      ~MarketAwareIndDecentralizingAgent.respond_to_negotiation_request
      ~MarketAwareIndDecentralizingAgent.respond_to_negotiation_request_
      ~MarketAwareIndDecentralizingAgent.respond_to_renegotiation_request
      ~MarketAwareIndDecentralizingAgent.set_preferences
      ~MarketAwareIndDecentralizingAgent.set_renegotiation_agenda
      ~MarketAwareIndDecentralizingAgent.sign_all_contracts
      ~MarketAwareIndDecentralizingAgent.sign_contract
      ~MarketAwareIndDecentralizingAgent.spawn
      ~MarketAwareIndDecentralizingAgent.spawn_object
      ~MarketAwareIndDecentralizingAgent.start_negotiations
      ~MarketAwareIndDecentralizingAgent.step
      ~MarketAwareIndDecentralizingAgent.step_
      ~MarketAwareIndDecentralizingAgent.target_quantities
      ~MarketAwareIndDecentralizingAgent.target_quantity
      ~MarketAwareIndDecentralizingAgent.to_dict
      ~MarketAwareIndDecentralizingAgent.trade_prediction_before_step
      ~MarketAwareIndDecentralizingAgent.trade_prediction_init
      ~MarketAwareIndDecentralizingAgent.trade_prediction_step

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
   .. automethod:: can_be_produced
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
   .. automethod:: start_negotiations
   .. automethod:: step
   .. automethod:: step_
   .. automethod:: target_quantities
   .. automethod:: target_quantity
   .. automethod:: to_dict
   .. automethod:: trade_prediction_before_step
   .. automethod:: trade_prediction_init
   .. automethod:: trade_prediction_step
