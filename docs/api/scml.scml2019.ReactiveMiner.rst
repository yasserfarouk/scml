ReactiveMiner
=============

.. currentmodule:: scml.scml2019

.. autoclass:: ReactiveMiner
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~ReactiveMiner.accepted_negotiation_requests
      ~ReactiveMiner.awi
      ~ReactiveMiner.id
      ~ReactiveMiner.initialized
      ~ReactiveMiner.name
      ~ReactiveMiner.negotiation_requests
      ~ReactiveMiner.requested_negotiations
      ~ReactiveMiner.running_negotiations
      ~ReactiveMiner.short_type_name
      ~ReactiveMiner.type_name
      ~ReactiveMiner.unsigned_contracts
      ~ReactiveMiner.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~ReactiveMiner.can_expect_agreement
      ~ReactiveMiner.checkpoint
      ~ReactiveMiner.checkpoint_info
      ~ReactiveMiner.confirm_contract_execution
      ~ReactiveMiner.confirm_loan
      ~ReactiveMiner.confirm_partial_execution
      ~ReactiveMiner.create
      ~ReactiveMiner.create_negotiation_request
      ~ReactiveMiner.from_checkpoint
      ~ReactiveMiner.from_config
      ~ReactiveMiner.init
      ~ReactiveMiner.init_
      ~ReactiveMiner.notify
      ~ReactiveMiner.on_agent_bankrupt
      ~ReactiveMiner.on_cash_transfer
      ~ReactiveMiner.on_contract_breached
      ~ReactiveMiner.on_contract_cancelled
      ~ReactiveMiner.on_contract_cancelled_
      ~ReactiveMiner.on_contract_executed
      ~ReactiveMiner.on_contract_nullified
      ~ReactiveMiner.on_contract_signed
      ~ReactiveMiner.on_contract_signed_
      ~ReactiveMiner.on_contracts_finalized
      ~ReactiveMiner.on_event
      ~ReactiveMiner.on_inventory_change
      ~ReactiveMiner.on_neg_request_accepted
      ~ReactiveMiner.on_neg_request_accepted_
      ~ReactiveMiner.on_neg_request_rejected
      ~ReactiveMiner.on_neg_request_rejected_
      ~ReactiveMiner.on_negotiation_failure
      ~ReactiveMiner.on_negotiation_failure_
      ~ReactiveMiner.on_negotiation_success
      ~ReactiveMiner.on_negotiation_success_
      ~ReactiveMiner.on_new_cfp
      ~ReactiveMiner.on_new_report
      ~ReactiveMiner.on_remove_cfp
      ~ReactiveMiner.read_config
      ~ReactiveMiner.request_negotiation
      ~ReactiveMiner.respond_to_negotiation_request
      ~ReactiveMiner.respond_to_negotiation_request_
      ~ReactiveMiner.respond_to_renegotiation_request
      ~ReactiveMiner.set_profiles
      ~ReactiveMiner.set_renegotiation_agenda
      ~ReactiveMiner.sign_all_contracts
      ~ReactiveMiner.sign_contract
      ~ReactiveMiner.step
      ~ReactiveMiner.step_

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
   .. automethod:: set_profiles
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: step
   .. automethod:: step_
