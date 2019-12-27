DefaultGreedyManager
====================

.. currentmodule:: scml.scml2019

.. autoclass:: DefaultGreedyManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DefaultGreedyManager.accepted_negotiation_requests
      ~DefaultGreedyManager.awi
      ~DefaultGreedyManager.id
      ~DefaultGreedyManager.initialized
      ~DefaultGreedyManager.name
      ~DefaultGreedyManager.negotiation_requests
      ~DefaultGreedyManager.requested_negotiations
      ~DefaultGreedyManager.running_negotiations
      ~DefaultGreedyManager.short_type_name
      ~DefaultGreedyManager.type_name
      ~DefaultGreedyManager.unsigned_contracts
      ~DefaultGreedyManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~DefaultGreedyManager.can_expect_agreement
      ~DefaultGreedyManager.can_produce
      ~DefaultGreedyManager.can_secure_needs
      ~DefaultGreedyManager.checkpoint
      ~DefaultGreedyManager.checkpoint_info
      ~DefaultGreedyManager.confirm_contract_execution
      ~DefaultGreedyManager.confirm_loan
      ~DefaultGreedyManager.confirm_partial_execution
      ~DefaultGreedyManager.create
      ~DefaultGreedyManager.create_negotiation_request
      ~DefaultGreedyManager.from_checkpoint
      ~DefaultGreedyManager.from_config
      ~DefaultGreedyManager.init
      ~DefaultGreedyManager.init_
      ~DefaultGreedyManager.notify
      ~DefaultGreedyManager.on_agent_bankrupt
      ~DefaultGreedyManager.on_cash_transfer
      ~DefaultGreedyManager.on_contract_breached
      ~DefaultGreedyManager.on_contract_cancelled
      ~DefaultGreedyManager.on_contract_cancelled_
      ~DefaultGreedyManager.on_contract_executed
      ~DefaultGreedyManager.on_contract_nullified
      ~DefaultGreedyManager.on_contract_signed
      ~DefaultGreedyManager.on_contract_signed_
      ~DefaultGreedyManager.on_contracts_finalized
      ~DefaultGreedyManager.on_event
      ~DefaultGreedyManager.on_inventory_change
      ~DefaultGreedyManager.on_neg_request_accepted
      ~DefaultGreedyManager.on_neg_request_accepted_
      ~DefaultGreedyManager.on_neg_request_rejected
      ~DefaultGreedyManager.on_neg_request_rejected_
      ~DefaultGreedyManager.on_negotiation_failure
      ~DefaultGreedyManager.on_negotiation_failure_
      ~DefaultGreedyManager.on_negotiation_success
      ~DefaultGreedyManager.on_negotiation_success_
      ~DefaultGreedyManager.on_new_cfp
      ~DefaultGreedyManager.on_new_report
      ~DefaultGreedyManager.on_production_failure
      ~DefaultGreedyManager.on_production_success
      ~DefaultGreedyManager.on_remove_cfp
      ~DefaultGreedyManager.read_config
      ~DefaultGreedyManager.request_negotiation
      ~DefaultGreedyManager.respond_to_negotiation_request
      ~DefaultGreedyManager.respond_to_negotiation_request_
      ~DefaultGreedyManager.respond_to_renegotiation_request
      ~DefaultGreedyManager.set_renegotiation_agenda
      ~DefaultGreedyManager.sign_all_contracts
      ~DefaultGreedyManager.sign_contract
      ~DefaultGreedyManager.step
      ~DefaultGreedyManager.step_
      ~DefaultGreedyManager.total_utility

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: id
   .. autoattribute:: initialized
   .. autoattribute:: name
   .. autoattribute:: negotiation_requests
   .. autoattribute:: requested_negotiations
   .. autoattribute:: running_negotiations
   .. autoattribute:: short_type_name
   .. autoattribute:: type_name
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
   .. automethod:: on_production_failure
   .. automethod:: on_production_success
   .. automethod:: on_remove_cfp
   .. automethod:: read_config
   .. automethod:: request_negotiation
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: step
   .. automethod:: step_
   .. automethod:: total_utility
