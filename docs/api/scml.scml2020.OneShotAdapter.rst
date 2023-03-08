OneShotAdapter
==============

.. currentmodule:: scml.scml2020

.. autoclass:: OneShotAdapter
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotAdapter.accepted_negotiation_requests
      ~OneShotAdapter.adapted_object
      ~OneShotAdapter.awi
      ~OneShotAdapter.crisp_ufun
      ~OneShotAdapter.has_cardinal_preferences
      ~OneShotAdapter.has_preferences
      ~OneShotAdapter.has_ufun
      ~OneShotAdapter.id
      ~OneShotAdapter.initialized
      ~OneShotAdapter.internal_state
      ~OneShotAdapter.name
      ~OneShotAdapter.negotiation_requests
      ~OneShotAdapter.preferences
      ~OneShotAdapter.price_multiplier
      ~OneShotAdapter.prob_ufun
      ~OneShotAdapter.requested_negotiations
      ~OneShotAdapter.reserved_outcome
      ~OneShotAdapter.reserved_value
      ~OneShotAdapter.running_negotiations
      ~OneShotAdapter.short_type_name
      ~OneShotAdapter.type_name
      ~OneShotAdapter.type_postfix
      ~OneShotAdapter.ufun
      ~OneShotAdapter.unsigned_contracts
      ~OneShotAdapter.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotAdapter.before_step
      ~OneShotAdapter.can_be_produced
      ~OneShotAdapter.checkpoint
      ~OneShotAdapter.checkpoint_info
      ~OneShotAdapter.confirm_production
      ~OneShotAdapter.create
      ~OneShotAdapter.create_negotiation_request
      ~OneShotAdapter.from_checkpoint
      ~OneShotAdapter.from_config
      ~OneShotAdapter.get_current_balance
      ~OneShotAdapter.get_disposal_cost
      ~OneShotAdapter.get_disposal_cost_dev
      ~OneShotAdapter.get_disposal_cost_mean
      ~OneShotAdapter.get_exogenous_input
      ~OneShotAdapter.get_exogenous_output
      ~OneShotAdapter.get_profile
      ~OneShotAdapter.get_shortfall_penalty
      ~OneShotAdapter.get_shortfall_penalty_dev
      ~OneShotAdapter.get_shortfall_penalty_mean
      ~OneShotAdapter.init
      ~OneShotAdapter.init_
      ~OneShotAdapter.make_ufun
      ~OneShotAdapter.notify
      ~OneShotAdapter.on_agent_bankrupt
      ~OneShotAdapter.on_contract_breached
      ~OneShotAdapter.on_contract_cancelled
      ~OneShotAdapter.on_contract_cancelled_
      ~OneShotAdapter.on_contract_executed
      ~OneShotAdapter.on_contract_signed
      ~OneShotAdapter.on_contract_signed_
      ~OneShotAdapter.on_contracts_finalized
      ~OneShotAdapter.on_event
      ~OneShotAdapter.on_failures
      ~OneShotAdapter.on_neg_request_accepted
      ~OneShotAdapter.on_neg_request_accepted_
      ~OneShotAdapter.on_neg_request_rejected
      ~OneShotAdapter.on_neg_request_rejected_
      ~OneShotAdapter.on_negotiation_failure
      ~OneShotAdapter.on_negotiation_failure_
      ~OneShotAdapter.on_negotiation_success
      ~OneShotAdapter.on_negotiation_success_
      ~OneShotAdapter.on_preferences_changed
      ~OneShotAdapter.on_simulation_step_ended
      ~OneShotAdapter.on_simulation_step_started
      ~OneShotAdapter.read_config
      ~OneShotAdapter.respond_to_negotiation_request
      ~OneShotAdapter.respond_to_negotiation_request_
      ~OneShotAdapter.respond_to_renegotiation_request
      ~OneShotAdapter.set_preferences
      ~OneShotAdapter.set_renegotiation_agenda
      ~OneShotAdapter.sign_all_contracts
      ~OneShotAdapter.sign_contract
      ~OneShotAdapter.spawn
      ~OneShotAdapter.spawn_object
      ~OneShotAdapter.step
      ~OneShotAdapter.step_
      ~OneShotAdapter.to_dict
      ~OneShotAdapter.trade_prediction_before_step
      ~OneShotAdapter.trade_prediction_init
      ~OneShotAdapter.trade_prediction_step

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: adapted_object
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
   .. autoattribute:: price_multiplier
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
   .. automethod:: get_current_balance
   .. automethod:: get_disposal_cost
   .. automethod:: get_disposal_cost_dev
   .. automethod:: get_disposal_cost_mean
   .. automethod:: get_exogenous_input
   .. automethod:: get_exogenous_output
   .. automethod:: get_profile
   .. automethod:: get_shortfall_penalty
   .. automethod:: get_shortfall_penalty_dev
   .. automethod:: get_shortfall_penalty_mean
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: make_ufun
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
   .. automethod:: step
   .. automethod:: step_
   .. automethod:: to_dict
   .. automethod:: trade_prediction_before_step
   .. automethod:: trade_prediction_init
   .. automethod:: trade_prediction_step
