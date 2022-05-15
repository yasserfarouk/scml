IndependentNegotiationsAgent
============================

.. currentmodule:: scml.scml2020

.. autoclass:: IndependentNegotiationsAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~IndependentNegotiationsAgent.accepted_negotiation_requests
      ~IndependentNegotiationsAgent.awi
      ~IndependentNegotiationsAgent.crisp_ufun
      ~IndependentNegotiationsAgent.has_cardinal_preferences
      ~IndependentNegotiationsAgent.has_preferences
      ~IndependentNegotiationsAgent.id
      ~IndependentNegotiationsAgent.initialized
      ~IndependentNegotiationsAgent.internal_state
      ~IndependentNegotiationsAgent.name
      ~IndependentNegotiationsAgent.negotiation_requests
      ~IndependentNegotiationsAgent.preferences
      ~IndependentNegotiationsAgent.prob_ufun
      ~IndependentNegotiationsAgent.requested_negotiations
      ~IndependentNegotiationsAgent.reserved_outcome
      ~IndependentNegotiationsAgent.reserved_value
      ~IndependentNegotiationsAgent.running_negotiations
      ~IndependentNegotiationsAgent.short_type_name
      ~IndependentNegotiationsAgent.type_name
      ~IndependentNegotiationsAgent.type_postfix
      ~IndependentNegotiationsAgent.ufun
      ~IndependentNegotiationsAgent.unsigned_contracts
      ~IndependentNegotiationsAgent.use_trading
      ~IndependentNegotiationsAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~IndependentNegotiationsAgent.acceptable_unit_price
      ~IndependentNegotiationsAgent.before_step
      ~IndependentNegotiationsAgent.checkpoint
      ~IndependentNegotiationsAgent.checkpoint_info
      ~IndependentNegotiationsAgent.confirm_production
      ~IndependentNegotiationsAgent.create
      ~IndependentNegotiationsAgent.create_negotiation_request
      ~IndependentNegotiationsAgent.create_ufun
      ~IndependentNegotiationsAgent.from_checkpoint
      ~IndependentNegotiationsAgent.from_config
      ~IndependentNegotiationsAgent.init
      ~IndependentNegotiationsAgent.init_
      ~IndependentNegotiationsAgent.negotiator
      ~IndependentNegotiationsAgent.notify
      ~IndependentNegotiationsAgent.on_agent_bankrupt
      ~IndependentNegotiationsAgent.on_contract_breached
      ~IndependentNegotiationsAgent.on_contract_cancelled
      ~IndependentNegotiationsAgent.on_contract_cancelled_
      ~IndependentNegotiationsAgent.on_contract_executed
      ~IndependentNegotiationsAgent.on_contract_signed
      ~IndependentNegotiationsAgent.on_contract_signed_
      ~IndependentNegotiationsAgent.on_contracts_finalized
      ~IndependentNegotiationsAgent.on_event
      ~IndependentNegotiationsAgent.on_failures
      ~IndependentNegotiationsAgent.on_neg_request_accepted
      ~IndependentNegotiationsAgent.on_neg_request_accepted_
      ~IndependentNegotiationsAgent.on_neg_request_rejected
      ~IndependentNegotiationsAgent.on_neg_request_rejected_
      ~IndependentNegotiationsAgent.on_negotiation_failure
      ~IndependentNegotiationsAgent.on_negotiation_failure_
      ~IndependentNegotiationsAgent.on_negotiation_success
      ~IndependentNegotiationsAgent.on_negotiation_success_
      ~IndependentNegotiationsAgent.on_preferences_changed
      ~IndependentNegotiationsAgent.on_simulation_step_ended
      ~IndependentNegotiationsAgent.on_simulation_step_started
      ~IndependentNegotiationsAgent.read_config
      ~IndependentNegotiationsAgent.respond_to_negotiation_request
      ~IndependentNegotiationsAgent.respond_to_negotiation_request_
      ~IndependentNegotiationsAgent.respond_to_renegotiation_request
      ~IndependentNegotiationsAgent.set_preferences
      ~IndependentNegotiationsAgent.set_renegotiation_agenda
      ~IndependentNegotiationsAgent.sign_all_contracts
      ~IndependentNegotiationsAgent.sign_contract
      ~IndependentNegotiationsAgent.spawn
      ~IndependentNegotiationsAgent.spawn_object
      ~IndependentNegotiationsAgent.start_negotiations
      ~IndependentNegotiationsAgent.step
      ~IndependentNegotiationsAgent.step_
      ~IndependentNegotiationsAgent.target_quantities
      ~IndependentNegotiationsAgent.target_quantity
      ~IndependentNegotiationsAgent.to_dict
      ~IndependentNegotiationsAgent.trade_prediction_before_step
      ~IndependentNegotiationsAgent.trade_prediction_init
      ~IndependentNegotiationsAgent.trade_prediction_step

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
