MarketAwareReactiveAgent
========================

.. currentmodule:: scml.scml2020

.. autoclass:: MarketAwareReactiveAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MarketAwareReactiveAgent.accepted_negotiation_requests
      ~MarketAwareReactiveAgent.awi
      ~MarketAwareReactiveAgent.crisp_ufun
      ~MarketAwareReactiveAgent.has_cardinal_preferences
      ~MarketAwareReactiveAgent.has_preferences
      ~MarketAwareReactiveAgent.has_ufun
      ~MarketAwareReactiveAgent.id
      ~MarketAwareReactiveAgent.initialized
      ~MarketAwareReactiveAgent.internal_state
      ~MarketAwareReactiveAgent.name
      ~MarketAwareReactiveAgent.negotiation_requests
      ~MarketAwareReactiveAgent.preferences
      ~MarketAwareReactiveAgent.prob_ufun
      ~MarketAwareReactiveAgent.requested_negotiations
      ~MarketAwareReactiveAgent.reserved_outcome
      ~MarketAwareReactiveAgent.reserved_value
      ~MarketAwareReactiveAgent.running_negotiations
      ~MarketAwareReactiveAgent.short_type_name
      ~MarketAwareReactiveAgent.type_name
      ~MarketAwareReactiveAgent.type_postfix
      ~MarketAwareReactiveAgent.ufun
      ~MarketAwareReactiveAgent.unsigned_contracts
      ~MarketAwareReactiveAgent.use_trading
      ~MarketAwareReactiveAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MarketAwareReactiveAgent.acceptable_unit_price
      ~MarketAwareReactiveAgent.add_controller
      ~MarketAwareReactiveAgent.all_negotiations_concluded
      ~MarketAwareReactiveAgent.before_step
      ~MarketAwareReactiveAgent.can_be_produced
      ~MarketAwareReactiveAgent.checkpoint
      ~MarketAwareReactiveAgent.checkpoint_info
      ~MarketAwareReactiveAgent.confirm_production
      ~MarketAwareReactiveAgent.create
      ~MarketAwareReactiveAgent.create_controller
      ~MarketAwareReactiveAgent.create_negotiation_request
      ~MarketAwareReactiveAgent.from_checkpoint
      ~MarketAwareReactiveAgent.from_config
      ~MarketAwareReactiveAgent.init
      ~MarketAwareReactiveAgent.init_
      ~MarketAwareReactiveAgent.insert_controller
      ~MarketAwareReactiveAgent.notify
      ~MarketAwareReactiveAgent.on_agent_bankrupt
      ~MarketAwareReactiveAgent.on_contract_breached
      ~MarketAwareReactiveAgent.on_contract_cancelled
      ~MarketAwareReactiveAgent.on_contract_cancelled_
      ~MarketAwareReactiveAgent.on_contract_executed
      ~MarketAwareReactiveAgent.on_contract_signed
      ~MarketAwareReactiveAgent.on_contract_signed_
      ~MarketAwareReactiveAgent.on_contracts_finalized
      ~MarketAwareReactiveAgent.on_event
      ~MarketAwareReactiveAgent.on_failures
      ~MarketAwareReactiveAgent.on_neg_request_accepted
      ~MarketAwareReactiveAgent.on_neg_request_accepted_
      ~MarketAwareReactiveAgent.on_neg_request_rejected
      ~MarketAwareReactiveAgent.on_neg_request_rejected_
      ~MarketAwareReactiveAgent.on_negotiation_failure
      ~MarketAwareReactiveAgent.on_negotiation_failure_
      ~MarketAwareReactiveAgent.on_negotiation_success
      ~MarketAwareReactiveAgent.on_negotiation_success_
      ~MarketAwareReactiveAgent.on_preferences_changed
      ~MarketAwareReactiveAgent.on_simulation_step_ended
      ~MarketAwareReactiveAgent.on_simulation_step_started
      ~MarketAwareReactiveAgent.predict_quantity
      ~MarketAwareReactiveAgent.read_config
      ~MarketAwareReactiveAgent.respond_to_negotiation_request
      ~MarketAwareReactiveAgent.respond_to_negotiation_request_
      ~MarketAwareReactiveAgent.respond_to_renegotiation_request
      ~MarketAwareReactiveAgent.set_preferences
      ~MarketAwareReactiveAgent.set_renegotiation_agenda
      ~MarketAwareReactiveAgent.sign_all_contracts
      ~MarketAwareReactiveAgent.sign_contract
      ~MarketAwareReactiveAgent.spawn
      ~MarketAwareReactiveAgent.spawn_object
      ~MarketAwareReactiveAgent.start_negotiations
      ~MarketAwareReactiveAgent.step
      ~MarketAwareReactiveAgent.step_
      ~MarketAwareReactiveAgent.target_quantities
      ~MarketAwareReactiveAgent.target_quantity
      ~MarketAwareReactiveAgent.to_dict
      ~MarketAwareReactiveAgent.trade_prediction_before_step
      ~MarketAwareReactiveAgent.trade_prediction_init
      ~MarketAwareReactiveAgent.trade_prediction_step

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
