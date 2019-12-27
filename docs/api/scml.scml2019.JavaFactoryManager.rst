JavaFactoryManager
==================

.. currentmodule:: scml.scml2019

.. autoclass:: JavaFactoryManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~JavaFactoryManager.accepted_negotiation_requests
      ~JavaFactoryManager.awi
      ~JavaFactoryManager.id
      ~JavaFactoryManager.initialized
      ~JavaFactoryManager.name
      ~JavaFactoryManager.negotiation_requests
      ~JavaFactoryManager.requested_negotiations
      ~JavaFactoryManager.running_negotiations
      ~JavaFactoryManager.short_type_name
      ~JavaFactoryManager.type_name
      ~JavaFactoryManager.unsigned_contracts
      ~JavaFactoryManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~JavaFactoryManager.can_expect_agreement
      ~JavaFactoryManager.checkpoint
      ~JavaFactoryManager.checkpoint_info
      ~JavaFactoryManager.confirmContractExecution
      ~JavaFactoryManager.confirmLoan
      ~JavaFactoryManager.confirmPartialExecution
      ~JavaFactoryManager.confirm_contract_execution
      ~JavaFactoryManager.confirm_loan
      ~JavaFactoryManager.confirm_partial_execution
      ~JavaFactoryManager.create
      ~JavaFactoryManager.create_negotiation_request
      ~JavaFactoryManager.do_nothing_manager
      ~JavaFactoryManager.from_checkpoint
      ~JavaFactoryManager.from_config
      ~JavaFactoryManager.from_dict
      ~JavaFactoryManager.getCompiledProfiles
      ~JavaFactoryManager.getConsuming
      ~JavaFactoryManager.getContracts
      ~JavaFactoryManager.getID
      ~JavaFactoryManager.getLineProfiles
      ~JavaFactoryManager.getName
      ~JavaFactoryManager.getNegotiationRequests
      ~JavaFactoryManager.getProcesses
      ~JavaFactoryManager.getProducing
      ~JavaFactoryManager.getProducts
      ~JavaFactoryManager.getRequestedNegotiations
      ~JavaFactoryManager.getRunningNegotiations
      ~JavaFactoryManager.greedy_manager
      ~JavaFactoryManager.init
      ~JavaFactoryManager.initPython
      ~JavaFactoryManager.init_
      ~JavaFactoryManager.init_java_bridge
      ~JavaFactoryManager.notify
      ~JavaFactoryManager.onAgentBankrupt
      ~JavaFactoryManager.onCashTransfer
      ~JavaFactoryManager.onContractBreached
      ~JavaFactoryManager.onContractCancelled
      ~JavaFactoryManager.onContractExecuted
      ~JavaFactoryManager.onContractNullified
      ~JavaFactoryManager.onContractSigned
      ~JavaFactoryManager.onInventoryChange
      ~JavaFactoryManager.onNegRequestAccepted
      ~JavaFactoryManager.onNegRequestRejected
      ~JavaFactoryManager.onNegotiationFailure
      ~JavaFactoryManager.onNegotiationSuccess
      ~JavaFactoryManager.onNewCFP
      ~JavaFactoryManager.onNewReport
      ~JavaFactoryManager.onProductionFailure
      ~JavaFactoryManager.onProductionSuccess
      ~JavaFactoryManager.onRemoveCFP
      ~JavaFactoryManager.on_agent_bankrupt
      ~JavaFactoryManager.on_cash_transfer
      ~JavaFactoryManager.on_contract_breached
      ~JavaFactoryManager.on_contract_cancelled
      ~JavaFactoryManager.on_contract_cancelled_
      ~JavaFactoryManager.on_contract_executed
      ~JavaFactoryManager.on_contract_nullified
      ~JavaFactoryManager.on_contract_signed
      ~JavaFactoryManager.on_contract_signed_
      ~JavaFactoryManager.on_contracts_finalized
      ~JavaFactoryManager.on_event
      ~JavaFactoryManager.on_inventory_change
      ~JavaFactoryManager.on_neg_request_accepted
      ~JavaFactoryManager.on_neg_request_accepted_
      ~JavaFactoryManager.on_neg_request_rejected
      ~JavaFactoryManager.on_neg_request_rejected_
      ~JavaFactoryManager.on_negotiation_failure
      ~JavaFactoryManager.on_negotiation_failure_
      ~JavaFactoryManager.on_negotiation_success
      ~JavaFactoryManager.on_negotiation_success_
      ~JavaFactoryManager.on_new_cfp
      ~JavaFactoryManager.on_new_report
      ~JavaFactoryManager.on_production_failure
      ~JavaFactoryManager.on_production_success
      ~JavaFactoryManager.on_remove_cfp
      ~JavaFactoryManager.read_config
      ~JavaFactoryManager.requestNegotiation
      ~JavaFactoryManager.request_negotiation
      ~JavaFactoryManager.respondToNegotiationRequest
      ~JavaFactoryManager.respondToRenegotiationRequest
      ~JavaFactoryManager.respond_to_negotiation_request
      ~JavaFactoryManager.respond_to_negotiation_request_
      ~JavaFactoryManager.respond_to_renegotiation_request
      ~JavaFactoryManager.setID
      ~JavaFactoryManager.setName
      ~JavaFactoryManager.setRenegotiationAgenda
      ~JavaFactoryManager.set_renegotiation_agenda
      ~JavaFactoryManager.signContract
      ~JavaFactoryManager.sign_all_contracts
      ~JavaFactoryManager.sign_contract
      ~JavaFactoryManager.step
      ~JavaFactoryManager.stepPython
      ~JavaFactoryManager.step_

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
   .. automethod:: confirmContractExecution
   .. automethod:: confirmLoan
   .. automethod:: confirmPartialExecution
   .. automethod:: confirm_contract_execution
   .. automethod:: confirm_loan
   .. automethod:: confirm_partial_execution
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: do_nothing_manager
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: from_dict
   .. automethod:: getCompiledProfiles
   .. automethod:: getConsuming
   .. automethod:: getContracts
   .. automethod:: getID
   .. automethod:: getLineProfiles
   .. automethod:: getName
   .. automethod:: getNegotiationRequests
   .. automethod:: getProcesses
   .. automethod:: getProducing
   .. automethod:: getProducts
   .. automethod:: getRequestedNegotiations
   .. automethod:: getRunningNegotiations
   .. automethod:: greedy_manager
   .. automethod:: init
   .. automethod:: initPython
   .. automethod:: init_
   .. automethod:: init_java_bridge
   .. automethod:: notify
   .. automethod:: onAgentBankrupt
   .. automethod:: onCashTransfer
   .. automethod:: onContractBreached
   .. automethod:: onContractCancelled
   .. automethod:: onContractExecuted
   .. automethod:: onContractNullified
   .. automethod:: onContractSigned
   .. automethod:: onInventoryChange
   .. automethod:: onNegRequestAccepted
   .. automethod:: onNegRequestRejected
   .. automethod:: onNegotiationFailure
   .. automethod:: onNegotiationSuccess
   .. automethod:: onNewCFP
   .. automethod:: onNewReport
   .. automethod:: onProductionFailure
   .. automethod:: onProductionSuccess
   .. automethod:: onRemoveCFP
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
   .. automethod:: requestNegotiation
   .. automethod:: request_negotiation
   .. automethod:: respondToNegotiationRequest
   .. automethod:: respondToRenegotiationRequest
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: setID
   .. automethod:: setName
   .. automethod:: setRenegotiationAgenda
   .. automethod:: set_renegotiation_agenda
   .. automethod:: signContract
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: step
   .. automethod:: stepPython
   .. automethod:: step_
