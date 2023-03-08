RandomAgent
===========

.. currentmodule:: scml.scml2020

.. autoclass:: RandomAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~RandomAgent.accepted_negotiation_requests
      ~RandomAgent.awi
      ~RandomAgent.crisp_ufun
      ~RandomAgent.has_cardinal_preferences
      ~RandomAgent.has_preferences
      ~RandomAgent.has_ufun
      ~RandomAgent.id
      ~RandomAgent.initialized
      ~RandomAgent.internal_state
      ~RandomAgent.name
      ~RandomAgent.negotiation_requests
      ~RandomAgent.preferences
      ~RandomAgent.prob_ufun
      ~RandomAgent.requested_negotiations
      ~RandomAgent.reserved_outcome
      ~RandomAgent.reserved_value
      ~RandomAgent.running_negotiations
      ~RandomAgent.short_type_name
      ~RandomAgent.type_name
      ~RandomAgent.type_postfix
      ~RandomAgent.ufun
      ~RandomAgent.unsigned_contracts
      ~RandomAgent.use_trading
      ~RandomAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~RandomAgent.acceptable_unit_price
      ~RandomAgent.before_step
      ~RandomAgent.checkpoint
      ~RandomAgent.checkpoint_info
      ~RandomAgent.confirm_production
      ~RandomAgent.create
      ~RandomAgent.create_negotiation_request
      ~RandomAgent.create_ufun
      ~RandomAgent.from_checkpoint
      ~RandomAgent.from_config
      ~RandomAgent.init
      ~RandomAgent.init_
      ~RandomAgent.negotiator
      ~RandomAgent.notify
      ~RandomAgent.on_agent_bankrupt
      ~RandomAgent.on_contract_breached
      ~RandomAgent.on_contract_cancelled
      ~RandomAgent.on_contract_cancelled_
      ~RandomAgent.on_contract_executed
      ~RandomAgent.on_contract_signed
      ~RandomAgent.on_contract_signed_
      ~RandomAgent.on_contracts_finalized
      ~RandomAgent.on_event
      ~RandomAgent.on_failures
      ~RandomAgent.on_neg_request_accepted
      ~RandomAgent.on_neg_request_accepted_
      ~RandomAgent.on_neg_request_rejected
      ~RandomAgent.on_neg_request_rejected_
      ~RandomAgent.on_negotiation_failure
      ~RandomAgent.on_negotiation_failure_
      ~RandomAgent.on_negotiation_success
      ~RandomAgent.on_negotiation_success_
      ~RandomAgent.on_preferences_changed
      ~RandomAgent.on_simulation_step_ended
      ~RandomAgent.on_simulation_step_started
      ~RandomAgent.read_config
      ~RandomAgent.respond_to_negotiation_request
      ~RandomAgent.respond_to_negotiation_request_
      ~RandomAgent.respond_to_renegotiation_request
      ~RandomAgent.set_preferences
      ~RandomAgent.set_renegotiation_agenda
      ~RandomAgent.sign_all_contracts
      ~RandomAgent.sign_contract
      ~RandomAgent.spawn
      ~RandomAgent.spawn_object
      ~RandomAgent.start_negotiations
      ~RandomAgent.step
      ~RandomAgent.step_
      ~RandomAgent.target_quantities
      ~RandomAgent.target_quantity
      ~RandomAgent.to_dict
      ~RandomAgent.trade_prediction_before_step
      ~RandomAgent.trade_prediction_init
      ~RandomAgent.trade_prediction_step

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
