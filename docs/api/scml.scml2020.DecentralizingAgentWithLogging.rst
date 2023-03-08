DecentralizingAgentWithLogging
==============================

.. currentmodule:: scml.scml2020

.. autoclass:: DecentralizingAgentWithLogging
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DecentralizingAgentWithLogging.accepted_negotiation_requests
      ~DecentralizingAgentWithLogging.awi
      ~DecentralizingAgentWithLogging.crisp_ufun
      ~DecentralizingAgentWithLogging.has_cardinal_preferences
      ~DecentralizingAgentWithLogging.has_preferences
      ~DecentralizingAgentWithLogging.has_ufun
      ~DecentralizingAgentWithLogging.id
      ~DecentralizingAgentWithLogging.initialized
      ~DecentralizingAgentWithLogging.internal_state
      ~DecentralizingAgentWithLogging.name
      ~DecentralizingAgentWithLogging.negotiation_requests
      ~DecentralizingAgentWithLogging.preferences
      ~DecentralizingAgentWithLogging.prob_ufun
      ~DecentralizingAgentWithLogging.requested_negotiations
      ~DecentralizingAgentWithLogging.reserved_outcome
      ~DecentralizingAgentWithLogging.reserved_value
      ~DecentralizingAgentWithLogging.running_negotiations
      ~DecentralizingAgentWithLogging.short_type_name
      ~DecentralizingAgentWithLogging.type_name
      ~DecentralizingAgentWithLogging.type_postfix
      ~DecentralizingAgentWithLogging.ufun
      ~DecentralizingAgentWithLogging.unsigned_contracts
      ~DecentralizingAgentWithLogging.use_trading
      ~DecentralizingAgentWithLogging.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~DecentralizingAgentWithLogging.acceptable_unit_price
      ~DecentralizingAgentWithLogging.add_controller
      ~DecentralizingAgentWithLogging.all_negotiations_concluded
      ~DecentralizingAgentWithLogging.before_step
      ~DecentralizingAgentWithLogging.can_be_produced
      ~DecentralizingAgentWithLogging.checkpoint
      ~DecentralizingAgentWithLogging.checkpoint_info
      ~DecentralizingAgentWithLogging.confirm_production
      ~DecentralizingAgentWithLogging.create
      ~DecentralizingAgentWithLogging.create_controller
      ~DecentralizingAgentWithLogging.create_negotiation_request
      ~DecentralizingAgentWithLogging.from_checkpoint
      ~DecentralizingAgentWithLogging.from_config
      ~DecentralizingAgentWithLogging.init
      ~DecentralizingAgentWithLogging.init_
      ~DecentralizingAgentWithLogging.insert_controller
      ~DecentralizingAgentWithLogging.notify
      ~DecentralizingAgentWithLogging.on_agent_bankrupt
      ~DecentralizingAgentWithLogging.on_contract_breached
      ~DecentralizingAgentWithLogging.on_contract_cancelled
      ~DecentralizingAgentWithLogging.on_contract_cancelled_
      ~DecentralizingAgentWithLogging.on_contract_executed
      ~DecentralizingAgentWithLogging.on_contract_signed
      ~DecentralizingAgentWithLogging.on_contract_signed_
      ~DecentralizingAgentWithLogging.on_contracts_finalized
      ~DecentralizingAgentWithLogging.on_event
      ~DecentralizingAgentWithLogging.on_failures
      ~DecentralizingAgentWithLogging.on_neg_request_accepted
      ~DecentralizingAgentWithLogging.on_neg_request_accepted_
      ~DecentralizingAgentWithLogging.on_neg_request_rejected
      ~DecentralizingAgentWithLogging.on_neg_request_rejected_
      ~DecentralizingAgentWithLogging.on_negotiation_failure
      ~DecentralizingAgentWithLogging.on_negotiation_failure_
      ~DecentralizingAgentWithLogging.on_negotiation_success
      ~DecentralizingAgentWithLogging.on_negotiation_success_
      ~DecentralizingAgentWithLogging.on_preferences_changed
      ~DecentralizingAgentWithLogging.on_simulation_step_ended
      ~DecentralizingAgentWithLogging.on_simulation_step_started
      ~DecentralizingAgentWithLogging.predict_quantity
      ~DecentralizingAgentWithLogging.read_config
      ~DecentralizingAgentWithLogging.respond_to_negotiation_request
      ~DecentralizingAgentWithLogging.respond_to_negotiation_request_
      ~DecentralizingAgentWithLogging.respond_to_renegotiation_request
      ~DecentralizingAgentWithLogging.set_preferences
      ~DecentralizingAgentWithLogging.set_renegotiation_agenda
      ~DecentralizingAgentWithLogging.sign_all_contracts
      ~DecentralizingAgentWithLogging.sign_contract
      ~DecentralizingAgentWithLogging.spawn
      ~DecentralizingAgentWithLogging.spawn_object
      ~DecentralizingAgentWithLogging.start_negotiations
      ~DecentralizingAgentWithLogging.step
      ~DecentralizingAgentWithLogging.step_
      ~DecentralizingAgentWithLogging.target_quantities
      ~DecentralizingAgentWithLogging.target_quantity
      ~DecentralizingAgentWithLogging.to_dict
      ~DecentralizingAgentWithLogging.trade_prediction_before_step
      ~DecentralizingAgentWithLogging.trade_prediction_init
      ~DecentralizingAgentWithLogging.trade_prediction_step

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
   .. automethod:: add_controller
   .. automethod:: all_negotiations_concluded
   .. automethod:: before_step
   .. automethod:: can_be_produced
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_controller
   .. automethod:: create_negotiation_request
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: insert_controller
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
