SCMLAgent
=========

.. currentmodule:: scml.scml2019

.. autoclass:: SCMLAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCMLAgent.accepted_negotiation_requests
      ~SCMLAgent.awi
      ~SCMLAgent.id
      ~SCMLAgent.initialized
      ~SCMLAgent.name
      ~SCMLAgent.negotiation_requests
      ~SCMLAgent.requested_negotiations
      ~SCMLAgent.running_negotiations
      ~SCMLAgent.short_type_name
      ~SCMLAgent.type_name
      ~SCMLAgent.unsigned_contracts
      ~SCMLAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCMLAgent.can_expect_agreement
      ~SCMLAgent.checkpoint
      ~SCMLAgent.checkpoint_info
      ~SCMLAgent.confirm_contract_execution
      ~SCMLAgent.confirm_loan
      ~SCMLAgent.confirm_partial_execution
      ~SCMLAgent.create
      ~SCMLAgent.create_negotiation_request
      ~SCMLAgent.from_checkpoint
      ~SCMLAgent.from_config
      ~SCMLAgent.init
      ~SCMLAgent.init_
      ~SCMLAgent.notify
      ~SCMLAgent.on_agent_bankrupt
      ~SCMLAgent.on_cash_transfer
      ~SCMLAgent.on_contract_breached
      ~SCMLAgent.on_contract_cancelled
      ~SCMLAgent.on_contract_cancelled_
      ~SCMLAgent.on_contract_executed
      ~SCMLAgent.on_contract_nullified
      ~SCMLAgent.on_contract_signed
      ~SCMLAgent.on_contract_signed_
      ~SCMLAgent.on_contracts_finalized
      ~SCMLAgent.on_event
      ~SCMLAgent.on_inventory_change
      ~SCMLAgent.on_neg_request_accepted
      ~SCMLAgent.on_neg_request_accepted_
      ~SCMLAgent.on_neg_request_rejected
      ~SCMLAgent.on_neg_request_rejected_
      ~SCMLAgent.on_negotiation_failure
      ~SCMLAgent.on_negotiation_failure_
      ~SCMLAgent.on_negotiation_success
      ~SCMLAgent.on_negotiation_success_
      ~SCMLAgent.on_new_cfp
      ~SCMLAgent.on_new_report
      ~SCMLAgent.on_remove_cfp
      ~SCMLAgent.read_config
      ~SCMLAgent.request_negotiation
      ~SCMLAgent.respond_to_negotiation_request
      ~SCMLAgent.respond_to_negotiation_request_
      ~SCMLAgent.respond_to_renegotiation_request
      ~SCMLAgent.set_renegotiation_agenda
      ~SCMLAgent.sign_all_contracts
      ~SCMLAgent.sign_contract
      ~SCMLAgent.step
      ~SCMLAgent.step_

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
