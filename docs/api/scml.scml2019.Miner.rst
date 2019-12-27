Miner
=====

.. currentmodule:: scml.scml2019

.. autoclass:: Miner
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Miner.accepted_negotiation_requests
      ~Miner.awi
      ~Miner.id
      ~Miner.initialized
      ~Miner.name
      ~Miner.negotiation_requests
      ~Miner.requested_negotiations
      ~Miner.running_negotiations
      ~Miner.short_type_name
      ~Miner.type_name
      ~Miner.unsigned_contracts
      ~Miner.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~Miner.can_expect_agreement
      ~Miner.checkpoint
      ~Miner.checkpoint_info
      ~Miner.confirm_contract_execution
      ~Miner.confirm_loan
      ~Miner.confirm_partial_execution
      ~Miner.create
      ~Miner.create_negotiation_request
      ~Miner.from_checkpoint
      ~Miner.from_config
      ~Miner.init
      ~Miner.init_
      ~Miner.notify
      ~Miner.on_agent_bankrupt
      ~Miner.on_cash_transfer
      ~Miner.on_contract_breached
      ~Miner.on_contract_cancelled
      ~Miner.on_contract_cancelled_
      ~Miner.on_contract_executed
      ~Miner.on_contract_nullified
      ~Miner.on_contract_signed
      ~Miner.on_contract_signed_
      ~Miner.on_contracts_finalized
      ~Miner.on_event
      ~Miner.on_inventory_change
      ~Miner.on_neg_request_accepted
      ~Miner.on_neg_request_accepted_
      ~Miner.on_neg_request_rejected
      ~Miner.on_neg_request_rejected_
      ~Miner.on_negotiation_failure
      ~Miner.on_negotiation_failure_
      ~Miner.on_negotiation_success
      ~Miner.on_negotiation_success_
      ~Miner.on_new_cfp
      ~Miner.on_new_report
      ~Miner.on_remove_cfp
      ~Miner.read_config
      ~Miner.request_negotiation
      ~Miner.respond_to_negotiation_request
      ~Miner.respond_to_negotiation_request_
      ~Miner.respond_to_renegotiation_request
      ~Miner.set_renegotiation_agenda
      ~Miner.sign_all_contracts
      ~Miner.sign_contract
      ~Miner.step
      ~Miner.step_

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
