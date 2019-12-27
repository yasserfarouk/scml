JavaDummyMiddleMan
==================

.. currentmodule:: scml.scml2019

.. autoclass:: JavaDummyMiddleMan
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~JavaDummyMiddleMan.accepted_negotiation_requests
      ~JavaDummyMiddleMan.awi
      ~JavaDummyMiddleMan.id
      ~JavaDummyMiddleMan.initialized
      ~JavaDummyMiddleMan.name
      ~JavaDummyMiddleMan.negotiation_requests
      ~JavaDummyMiddleMan.requested_negotiations
      ~JavaDummyMiddleMan.running_negotiations
      ~JavaDummyMiddleMan.short_type_name
      ~JavaDummyMiddleMan.type_name
      ~JavaDummyMiddleMan.unsigned_contracts
      ~JavaDummyMiddleMan.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~JavaDummyMiddleMan.can_expect_agreement
      ~JavaDummyMiddleMan.checkpoint
      ~JavaDummyMiddleMan.checkpoint_info
      ~JavaDummyMiddleMan.confirmContractExecution
      ~JavaDummyMiddleMan.confirmLoan
      ~JavaDummyMiddleMan.confirmPartialExecution
      ~JavaDummyMiddleMan.confirm_contract_execution
      ~JavaDummyMiddleMan.confirm_loan
      ~JavaDummyMiddleMan.confirm_partial_execution
      ~JavaDummyMiddleMan.create
      ~JavaDummyMiddleMan.create_negotiation_request
      ~JavaDummyMiddleMan.do_nothing_manager
      ~JavaDummyMiddleMan.from_checkpoint
      ~JavaDummyMiddleMan.from_config
      ~JavaDummyMiddleMan.from_dict
      ~JavaDummyMiddleMan.getCompiledProfiles
      ~JavaDummyMiddleMan.getConsuming
      ~JavaDummyMiddleMan.getContracts
      ~JavaDummyMiddleMan.getID
      ~JavaDummyMiddleMan.getLineProfiles
      ~JavaDummyMiddleMan.getName
      ~JavaDummyMiddleMan.getNegotiationRequests
      ~JavaDummyMiddleMan.getProcesses
      ~JavaDummyMiddleMan.getProducing
      ~JavaDummyMiddleMan.getProducts
      ~JavaDummyMiddleMan.getRequestedNegotiations
      ~JavaDummyMiddleMan.getRunningNegotiations
      ~JavaDummyMiddleMan.greedy_manager
      ~JavaDummyMiddleMan.init
      ~JavaDummyMiddleMan.initPython
      ~JavaDummyMiddleMan.init_
      ~JavaDummyMiddleMan.init_java_bridge
      ~JavaDummyMiddleMan.notify
      ~JavaDummyMiddleMan.onAgentBankrupt
      ~JavaDummyMiddleMan.onCashTransfer
      ~JavaDummyMiddleMan.onContractBreached
      ~JavaDummyMiddleMan.onContractCancelled
      ~JavaDummyMiddleMan.onContractExecuted
      ~JavaDummyMiddleMan.onContractNullified
      ~JavaDummyMiddleMan.onContractSigned
      ~JavaDummyMiddleMan.onInventoryChange
      ~JavaDummyMiddleMan.onNegRequestAccepted
      ~JavaDummyMiddleMan.onNegRequestRejected
      ~JavaDummyMiddleMan.onNegotiationFailure
      ~JavaDummyMiddleMan.onNegotiationSuccess
      ~JavaDummyMiddleMan.onNewCFP
      ~JavaDummyMiddleMan.onNewReport
      ~JavaDummyMiddleMan.onProductionFailure
      ~JavaDummyMiddleMan.onProductionSuccess
      ~JavaDummyMiddleMan.onRemoveCFP
      ~JavaDummyMiddleMan.on_agent_bankrupt
      ~JavaDummyMiddleMan.on_cash_transfer
      ~JavaDummyMiddleMan.on_contract_breached
      ~JavaDummyMiddleMan.on_contract_cancelled
      ~JavaDummyMiddleMan.on_contract_cancelled_
      ~JavaDummyMiddleMan.on_contract_executed
      ~JavaDummyMiddleMan.on_contract_nullified
      ~JavaDummyMiddleMan.on_contract_signed
      ~JavaDummyMiddleMan.on_contract_signed_
      ~JavaDummyMiddleMan.on_contracts_finalized
      ~JavaDummyMiddleMan.on_event
      ~JavaDummyMiddleMan.on_inventory_change
      ~JavaDummyMiddleMan.on_neg_request_accepted
      ~JavaDummyMiddleMan.on_neg_request_accepted_
      ~JavaDummyMiddleMan.on_neg_request_rejected
      ~JavaDummyMiddleMan.on_neg_request_rejected_
      ~JavaDummyMiddleMan.on_negotiation_failure
      ~JavaDummyMiddleMan.on_negotiation_failure_
      ~JavaDummyMiddleMan.on_negotiation_success
      ~JavaDummyMiddleMan.on_negotiation_success_
      ~JavaDummyMiddleMan.on_new_cfp
      ~JavaDummyMiddleMan.on_new_report
      ~JavaDummyMiddleMan.on_production_failure
      ~JavaDummyMiddleMan.on_production_success
      ~JavaDummyMiddleMan.on_remove_cfp
      ~JavaDummyMiddleMan.read_config
      ~JavaDummyMiddleMan.requestNegotiation
      ~JavaDummyMiddleMan.request_negotiation
      ~JavaDummyMiddleMan.respondToNegotiationRequest
      ~JavaDummyMiddleMan.respondToRenegotiationRequest
      ~JavaDummyMiddleMan.respond_to_negotiation_request
      ~JavaDummyMiddleMan.respond_to_negotiation_request_
      ~JavaDummyMiddleMan.respond_to_renegotiation_request
      ~JavaDummyMiddleMan.setID
      ~JavaDummyMiddleMan.setName
      ~JavaDummyMiddleMan.setRenegotiationAgenda
      ~JavaDummyMiddleMan.set_renegotiation_agenda
      ~JavaDummyMiddleMan.signContract
      ~JavaDummyMiddleMan.sign_all_contracts
      ~JavaDummyMiddleMan.sign_contract
      ~JavaDummyMiddleMan.step
      ~JavaDummyMiddleMan.stepPython
      ~JavaDummyMiddleMan.step_

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
