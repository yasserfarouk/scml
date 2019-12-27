JavaGreedyFactoryManager
========================

.. currentmodule:: scml.scml2019

.. autoclass:: JavaGreedyFactoryManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~JavaGreedyFactoryManager.accepted_negotiation_requests
      ~JavaGreedyFactoryManager.awi
      ~JavaGreedyFactoryManager.id
      ~JavaGreedyFactoryManager.initialized
      ~JavaGreedyFactoryManager.name
      ~JavaGreedyFactoryManager.negotiation_requests
      ~JavaGreedyFactoryManager.requested_negotiations
      ~JavaGreedyFactoryManager.running_negotiations
      ~JavaGreedyFactoryManager.short_type_name
      ~JavaGreedyFactoryManager.type_name
      ~JavaGreedyFactoryManager.unsigned_contracts
      ~JavaGreedyFactoryManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~JavaGreedyFactoryManager.can_expect_agreement
      ~JavaGreedyFactoryManager.checkpoint
      ~JavaGreedyFactoryManager.checkpoint_info
      ~JavaGreedyFactoryManager.confirmContractExecution
      ~JavaGreedyFactoryManager.confirmLoan
      ~JavaGreedyFactoryManager.confirmPartialExecution
      ~JavaGreedyFactoryManager.confirm_contract_execution
      ~JavaGreedyFactoryManager.confirm_loan
      ~JavaGreedyFactoryManager.confirm_partial_execution
      ~JavaGreedyFactoryManager.create
      ~JavaGreedyFactoryManager.create_negotiation_request
      ~JavaGreedyFactoryManager.do_nothing_manager
      ~JavaGreedyFactoryManager.from_checkpoint
      ~JavaGreedyFactoryManager.from_config
      ~JavaGreedyFactoryManager.from_dict
      ~JavaGreedyFactoryManager.getCompiledProfiles
      ~JavaGreedyFactoryManager.getConsuming
      ~JavaGreedyFactoryManager.getContracts
      ~JavaGreedyFactoryManager.getID
      ~JavaGreedyFactoryManager.getLineProfiles
      ~JavaGreedyFactoryManager.getName
      ~JavaGreedyFactoryManager.getNegotiationRequests
      ~JavaGreedyFactoryManager.getProcesses
      ~JavaGreedyFactoryManager.getProducing
      ~JavaGreedyFactoryManager.getProducts
      ~JavaGreedyFactoryManager.getRequestedNegotiations
      ~JavaGreedyFactoryManager.getRunningNegotiations
      ~JavaGreedyFactoryManager.greedy_manager
      ~JavaGreedyFactoryManager.init
      ~JavaGreedyFactoryManager.initPython
      ~JavaGreedyFactoryManager.init_
      ~JavaGreedyFactoryManager.init_java_bridge
      ~JavaGreedyFactoryManager.notify
      ~JavaGreedyFactoryManager.onAgentBankrupt
      ~JavaGreedyFactoryManager.onCashTransfer
      ~JavaGreedyFactoryManager.onContractBreached
      ~JavaGreedyFactoryManager.onContractCancelled
      ~JavaGreedyFactoryManager.onContractExecuted
      ~JavaGreedyFactoryManager.onContractNullified
      ~JavaGreedyFactoryManager.onContractSigned
      ~JavaGreedyFactoryManager.onInventoryChange
      ~JavaGreedyFactoryManager.onNegRequestAccepted
      ~JavaGreedyFactoryManager.onNegRequestRejected
      ~JavaGreedyFactoryManager.onNegotiationFailure
      ~JavaGreedyFactoryManager.onNegotiationSuccess
      ~JavaGreedyFactoryManager.onNewCFP
      ~JavaGreedyFactoryManager.onNewReport
      ~JavaGreedyFactoryManager.onProductionFailure
      ~JavaGreedyFactoryManager.onProductionSuccess
      ~JavaGreedyFactoryManager.onRemoveCFP
      ~JavaGreedyFactoryManager.on_agent_bankrupt
      ~JavaGreedyFactoryManager.on_cash_transfer
      ~JavaGreedyFactoryManager.on_contract_breached
      ~JavaGreedyFactoryManager.on_contract_cancelled
      ~JavaGreedyFactoryManager.on_contract_cancelled_
      ~JavaGreedyFactoryManager.on_contract_executed
      ~JavaGreedyFactoryManager.on_contract_nullified
      ~JavaGreedyFactoryManager.on_contract_signed
      ~JavaGreedyFactoryManager.on_contract_signed_
      ~JavaGreedyFactoryManager.on_contracts_finalized
      ~JavaGreedyFactoryManager.on_event
      ~JavaGreedyFactoryManager.on_inventory_change
      ~JavaGreedyFactoryManager.on_neg_request_accepted
      ~JavaGreedyFactoryManager.on_neg_request_accepted_
      ~JavaGreedyFactoryManager.on_neg_request_rejected
      ~JavaGreedyFactoryManager.on_neg_request_rejected_
      ~JavaGreedyFactoryManager.on_negotiation_failure
      ~JavaGreedyFactoryManager.on_negotiation_failure_
      ~JavaGreedyFactoryManager.on_negotiation_success
      ~JavaGreedyFactoryManager.on_negotiation_success_
      ~JavaGreedyFactoryManager.on_new_cfp
      ~JavaGreedyFactoryManager.on_new_report
      ~JavaGreedyFactoryManager.on_production_failure
      ~JavaGreedyFactoryManager.on_production_success
      ~JavaGreedyFactoryManager.on_remove_cfp
      ~JavaGreedyFactoryManager.read_config
      ~JavaGreedyFactoryManager.requestNegotiation
      ~JavaGreedyFactoryManager.request_negotiation
      ~JavaGreedyFactoryManager.respondToNegotiationRequest
      ~JavaGreedyFactoryManager.respondToRenegotiationRequest
      ~JavaGreedyFactoryManager.respond_to_negotiation_request
      ~JavaGreedyFactoryManager.respond_to_negotiation_request_
      ~JavaGreedyFactoryManager.respond_to_renegotiation_request
      ~JavaGreedyFactoryManager.setID
      ~JavaGreedyFactoryManager.setName
      ~JavaGreedyFactoryManager.setRenegotiationAgenda
      ~JavaGreedyFactoryManager.set_renegotiation_agenda
      ~JavaGreedyFactoryManager.signContract
      ~JavaGreedyFactoryManager.sign_all_contracts
      ~JavaGreedyFactoryManager.sign_contract
      ~JavaGreedyFactoryManager.step
      ~JavaGreedyFactoryManager.stepPython
      ~JavaGreedyFactoryManager.step_

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
