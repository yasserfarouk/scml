DoNothingFactoryManager
=======================

.. currentmodule:: scml.scml2019

.. autoclass:: DoNothingFactoryManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DoNothingFactoryManager.accepted_negotiation_requests
      ~DoNothingFactoryManager.awi
      ~DoNothingFactoryManager.crisp_ufun
      ~DoNothingFactoryManager.has_cardinal_preferences
      ~DoNothingFactoryManager.has_preferences
      ~DoNothingFactoryManager.id
      ~DoNothingFactoryManager.initialized
      ~DoNothingFactoryManager.name
      ~DoNothingFactoryManager.negotiation_requests
      ~DoNothingFactoryManager.preferences
      ~DoNothingFactoryManager.prob_ufun
      ~DoNothingFactoryManager.requested_negotiations
      ~DoNothingFactoryManager.reserved_outcome
      ~DoNothingFactoryManager.reserved_value
      ~DoNothingFactoryManager.running_negotiations
      ~DoNothingFactoryManager.short_type_name
      ~DoNothingFactoryManager.type_name
      ~DoNothingFactoryManager.type_postfix
      ~DoNothingFactoryManager.ufun
      ~DoNothingFactoryManager.unsigned_contracts
      ~DoNothingFactoryManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~DoNothingFactoryManager.can_expect_agreement
      ~DoNothingFactoryManager.checkpoint
      ~DoNothingFactoryManager.checkpoint_info
      ~DoNothingFactoryManager.confirm_contract_execution
      ~DoNothingFactoryManager.confirm_loan
      ~DoNothingFactoryManager.confirm_partial_execution
      ~DoNothingFactoryManager.create
      ~DoNothingFactoryManager.create_negotiation_request
      ~DoNothingFactoryManager.from_checkpoint
      ~DoNothingFactoryManager.from_config
      ~DoNothingFactoryManager.init
      ~DoNothingFactoryManager.init_
      ~DoNothingFactoryManager.notify
      ~DoNothingFactoryManager.on_agent_bankrupt
      ~DoNothingFactoryManager.on_cash_transfer
      ~DoNothingFactoryManager.on_contract_breached
      ~DoNothingFactoryManager.on_contract_cancelled
      ~DoNothingFactoryManager.on_contract_cancelled_
      ~DoNothingFactoryManager.on_contract_executed
      ~DoNothingFactoryManager.on_contract_nullified
      ~DoNothingFactoryManager.on_contract_signed
      ~DoNothingFactoryManager.on_contract_signed_
      ~DoNothingFactoryManager.on_contracts_finalized
      ~DoNothingFactoryManager.on_event
      ~DoNothingFactoryManager.on_inventory_change
      ~DoNothingFactoryManager.on_neg_request_accepted
      ~DoNothingFactoryManager.on_neg_request_accepted_
      ~DoNothingFactoryManager.on_neg_request_rejected
      ~DoNothingFactoryManager.on_neg_request_rejected_
      ~DoNothingFactoryManager.on_negotiation_failure
      ~DoNothingFactoryManager.on_negotiation_failure_
      ~DoNothingFactoryManager.on_negotiation_success
      ~DoNothingFactoryManager.on_negotiation_success_
      ~DoNothingFactoryManager.on_new_cfp
      ~DoNothingFactoryManager.on_new_report
      ~DoNothingFactoryManager.on_preferences_changed
      ~DoNothingFactoryManager.on_production_failure
      ~DoNothingFactoryManager.on_production_success
      ~DoNothingFactoryManager.on_remove_cfp
      ~DoNothingFactoryManager.on_simulation_step_ended
      ~DoNothingFactoryManager.on_simulation_step_started
      ~DoNothingFactoryManager.read_config
      ~DoNothingFactoryManager.request_negotiation
      ~DoNothingFactoryManager.respond_to_negotiation_request
      ~DoNothingFactoryManager.respond_to_negotiation_request_
      ~DoNothingFactoryManager.respond_to_renegotiation_request
      ~DoNothingFactoryManager.set_preferences
      ~DoNothingFactoryManager.set_renegotiation_agenda
      ~DoNothingFactoryManager.sign_all_contracts
      ~DoNothingFactoryManager.sign_contract
      ~DoNothingFactoryManager.spawn
      ~DoNothingFactoryManager.spawn_object
      ~DoNothingFactoryManager.step
      ~DoNothingFactoryManager.step_

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: id
   .. autoattribute:: initialized
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

   .. automethod:: can_expect_agreement
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_contract_execution
   .. automethod:: confirm_loan
   .. automethod:: confirm_partial_execution
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: notify
   .. automethod:: on_agent_bankrupt
   .. automethod:: on_cash_transfer
   .. automethod:: on_contract_breached
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_cancelled_
   .. automethod:: on_contract_executed
   .. automethod:: on_contract_nullified
   .. automethod:: on_contract_signed
   .. automethod:: on_contract_signed_
   .. automethod:: on_contracts_finalized
   .. automethod:: on_event
   .. automethod:: on_inventory_change
   .. automethod:: on_neg_request_accepted
   .. automethod:: on_neg_request_accepted_
   .. automethod:: on_neg_request_rejected
   .. automethod:: on_neg_request_rejected_
   .. automethod:: on_negotiation_failure
   .. automethod:: on_negotiation_failure_
   .. automethod:: on_negotiation_success
   .. automethod:: on_negotiation_success_
   .. automethod:: on_new_cfp
   .. automethod:: on_new_report
   .. automethod:: on_preferences_changed
   .. automethod:: on_production_failure
   .. automethod:: on_production_success
   .. automethod:: on_remove_cfp
   .. automethod:: on_simulation_step_ended
   .. automethod:: on_simulation_step_started
   .. automethod:: read_config
   .. automethod:: request_negotiation
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
