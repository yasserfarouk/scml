GreedyFactoryManager
====================

.. currentmodule:: scml.scml2019

.. autoclass:: GreedyFactoryManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~GreedyFactoryManager.accepted_negotiation_requests
      ~GreedyFactoryManager.awi
      ~GreedyFactoryManager.crisp_ufun
      ~GreedyFactoryManager.has_cardinal_preferences
      ~GreedyFactoryManager.has_preferences
      ~GreedyFactoryManager.has_ufun
      ~GreedyFactoryManager.id
      ~GreedyFactoryManager.initialized
      ~GreedyFactoryManager.name
      ~GreedyFactoryManager.negotiation_requests
      ~GreedyFactoryManager.preferences
      ~GreedyFactoryManager.prob_ufun
      ~GreedyFactoryManager.requested_negotiations
      ~GreedyFactoryManager.reserved_outcome
      ~GreedyFactoryManager.reserved_value
      ~GreedyFactoryManager.running_negotiations
      ~GreedyFactoryManager.short_type_name
      ~GreedyFactoryManager.type_name
      ~GreedyFactoryManager.type_postfix
      ~GreedyFactoryManager.ufun
      ~GreedyFactoryManager.unsigned_contracts
      ~GreedyFactoryManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~GreedyFactoryManager.can_expect_agreement
      ~GreedyFactoryManager.can_produce
      ~GreedyFactoryManager.can_secure_needs
      ~GreedyFactoryManager.checkpoint
      ~GreedyFactoryManager.checkpoint_info
      ~GreedyFactoryManager.confirm_contract_execution
      ~GreedyFactoryManager.confirm_loan
      ~GreedyFactoryManager.confirm_partial_execution
      ~GreedyFactoryManager.create
      ~GreedyFactoryManager.create_negotiation_request
      ~GreedyFactoryManager.from_checkpoint
      ~GreedyFactoryManager.from_config
      ~GreedyFactoryManager.init
      ~GreedyFactoryManager.init_
      ~GreedyFactoryManager.notify
      ~GreedyFactoryManager.on_agent_bankrupt
      ~GreedyFactoryManager.on_cash_transfer
      ~GreedyFactoryManager.on_contract_breached
      ~GreedyFactoryManager.on_contract_cancelled
      ~GreedyFactoryManager.on_contract_cancelled_
      ~GreedyFactoryManager.on_contract_executed
      ~GreedyFactoryManager.on_contract_nullified
      ~GreedyFactoryManager.on_contract_signed
      ~GreedyFactoryManager.on_contract_signed_
      ~GreedyFactoryManager.on_contracts_finalized
      ~GreedyFactoryManager.on_event
      ~GreedyFactoryManager.on_inventory_change
      ~GreedyFactoryManager.on_neg_request_accepted
      ~GreedyFactoryManager.on_neg_request_accepted_
      ~GreedyFactoryManager.on_neg_request_rejected
      ~GreedyFactoryManager.on_neg_request_rejected_
      ~GreedyFactoryManager.on_negotiation_failure
      ~GreedyFactoryManager.on_negotiation_failure_
      ~GreedyFactoryManager.on_negotiation_success
      ~GreedyFactoryManager.on_negotiation_success_
      ~GreedyFactoryManager.on_new_cfp
      ~GreedyFactoryManager.on_new_report
      ~GreedyFactoryManager.on_preferences_changed
      ~GreedyFactoryManager.on_production_failure
      ~GreedyFactoryManager.on_production_success
      ~GreedyFactoryManager.on_remove_cfp
      ~GreedyFactoryManager.on_simulation_step_ended
      ~GreedyFactoryManager.on_simulation_step_started
      ~GreedyFactoryManager.read_config
      ~GreedyFactoryManager.request_negotiation
      ~GreedyFactoryManager.respond_to_negotiation_request
      ~GreedyFactoryManager.respond_to_negotiation_request_
      ~GreedyFactoryManager.respond_to_renegotiation_request
      ~GreedyFactoryManager.set_preferences
      ~GreedyFactoryManager.set_renegotiation_agenda
      ~GreedyFactoryManager.sign_all_contracts
      ~GreedyFactoryManager.sign_contract
      ~GreedyFactoryManager.spawn
      ~GreedyFactoryManager.spawn_object
      ~GreedyFactoryManager.step
      ~GreedyFactoryManager.step_
      ~GreedyFactoryManager.total_utility

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: has_ufun
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
   .. automethod:: can_produce
   .. automethod:: can_secure_needs
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
   .. automethod:: total_utility
