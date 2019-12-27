FactoryManager
==============

.. currentmodule:: scml.scml2019

.. autoclass:: FactoryManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~FactoryManager.accepted_negotiation_requests
      ~FactoryManager.awi
      ~FactoryManager.id
      ~FactoryManager.initialized
      ~FactoryManager.name
      ~FactoryManager.negotiation_requests
      ~FactoryManager.requested_negotiations
      ~FactoryManager.running_negotiations
      ~FactoryManager.short_type_name
      ~FactoryManager.type_name
      ~FactoryManager.unsigned_contracts
      ~FactoryManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~FactoryManager.can_expect_agreement
      ~FactoryManager.checkpoint
      ~FactoryManager.checkpoint_info
      ~FactoryManager.confirm_contract_execution
      ~FactoryManager.confirm_loan
      ~FactoryManager.confirm_partial_execution
      ~FactoryManager.create
      ~FactoryManager.create_negotiation_request
      ~FactoryManager.from_checkpoint
      ~FactoryManager.from_config
      ~FactoryManager.init
      ~FactoryManager.init_
      ~FactoryManager.notify
      ~FactoryManager.on_agent_bankrupt
      ~FactoryManager.on_cash_transfer
      ~FactoryManager.on_contract_breached
      ~FactoryManager.on_contract_cancelled
      ~FactoryManager.on_contract_cancelled_
      ~FactoryManager.on_contract_executed
      ~FactoryManager.on_contract_nullified
      ~FactoryManager.on_contract_signed
      ~FactoryManager.on_contract_signed_
      ~FactoryManager.on_contracts_finalized
      ~FactoryManager.on_event
      ~FactoryManager.on_inventory_change
      ~FactoryManager.on_neg_request_accepted
      ~FactoryManager.on_neg_request_accepted_
      ~FactoryManager.on_neg_request_rejected
      ~FactoryManager.on_neg_request_rejected_
      ~FactoryManager.on_negotiation_failure
      ~FactoryManager.on_negotiation_failure_
      ~FactoryManager.on_negotiation_success
      ~FactoryManager.on_negotiation_success_
      ~FactoryManager.on_new_cfp
      ~FactoryManager.on_new_report
      ~FactoryManager.on_production_failure
      ~FactoryManager.on_production_success
      ~FactoryManager.on_remove_cfp
      ~FactoryManager.read_config
      ~FactoryManager.request_negotiation
      ~FactoryManager.respond_to_negotiation_request
      ~FactoryManager.respond_to_negotiation_request_
      ~FactoryManager.respond_to_renegotiation_request
      ~FactoryManager.set_renegotiation_agenda
      ~FactoryManager.sign_all_contracts
      ~FactoryManager.sign_contract
      ~FactoryManager.step
      ~FactoryManager.step_

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
