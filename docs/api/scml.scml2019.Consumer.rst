Consumer
========

.. currentmodule:: scml.scml2019

.. autoclass:: Consumer
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Consumer.accepted_negotiation_requests
      ~Consumer.awi
      ~Consumer.id
      ~Consumer.initialized
      ~Consumer.name
      ~Consumer.negotiation_requests
      ~Consumer.requested_negotiations
      ~Consumer.running_negotiations
      ~Consumer.short_type_name
      ~Consumer.type_name
      ~Consumer.unsigned_contracts
      ~Consumer.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~Consumer.can_expect_agreement
      ~Consumer.checkpoint
      ~Consumer.checkpoint_info
      ~Consumer.confirm_contract_execution
      ~Consumer.confirm_loan
      ~Consumer.confirm_partial_execution
      ~Consumer.create
      ~Consumer.create_negotiation_request
      ~Consumer.from_checkpoint
      ~Consumer.from_config
      ~Consumer.init
      ~Consumer.init_
      ~Consumer.notify
      ~Consumer.on_agent_bankrupt
      ~Consumer.on_cash_transfer
      ~Consumer.on_contract_breached
      ~Consumer.on_contract_cancelled
      ~Consumer.on_contract_cancelled_
      ~Consumer.on_contract_executed
      ~Consumer.on_contract_nullified
      ~Consumer.on_contract_signed
      ~Consumer.on_contract_signed_
      ~Consumer.on_contracts_finalized
      ~Consumer.on_event
      ~Consumer.on_inventory_change
      ~Consumer.on_neg_request_accepted
      ~Consumer.on_neg_request_accepted_
      ~Consumer.on_neg_request_rejected
      ~Consumer.on_neg_request_rejected_
      ~Consumer.on_negotiation_failure
      ~Consumer.on_negotiation_failure_
      ~Consumer.on_negotiation_success
      ~Consumer.on_negotiation_success_
      ~Consumer.on_new_cfp
      ~Consumer.on_new_report
      ~Consumer.on_remove_cfp
      ~Consumer.read_config
      ~Consumer.request_negotiation
      ~Consumer.respond_to_negotiation_request
      ~Consumer.respond_to_negotiation_request_
      ~Consumer.respond_to_renegotiation_request
      ~Consumer.set_renegotiation_agenda
      ~Consumer.sign_all_contracts
      ~Consumer.sign_contract
      ~Consumer.step
      ~Consumer.step_

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
