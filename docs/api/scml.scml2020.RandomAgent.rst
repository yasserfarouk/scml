RandomAgent
===========

.. currentmodule:: scml.scml2020

.. autoclass:: RandomAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~RandomAgent.accepted_negotiation_requests
      ~RandomAgent.awi
      ~RandomAgent.id
      ~RandomAgent.initialized
      ~RandomAgent.name
      ~RandomAgent.negotiation_requests
      ~RandomAgent.requested_negotiations
      ~RandomAgent.running_negotiations
      ~RandomAgent.short_type_name
      ~RandomAgent.type_name
      ~RandomAgent.unsigned_contracts
      ~RandomAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~RandomAgent.checkpoint
      ~RandomAgent.checkpoint_info
      ~RandomAgent.confirm_exogenous_sales
      ~RandomAgent.confirm_exogenous_supplies
      ~RandomAgent.confirm_production
      ~RandomAgent.create
      ~RandomAgent.create_negotiation_request
      ~RandomAgent.create_ufun
      ~RandomAgent.from_checkpoint
      ~RandomAgent.from_config
      ~RandomAgent.init
      ~RandomAgent.init_
      ~RandomAgent.negotiator
      ~RandomAgent.notify
      ~RandomAgent.on_contract_breached
      ~RandomAgent.on_contract_cancelled
      ~RandomAgent.on_contract_cancelled_
      ~RandomAgent.on_contract_executed
      ~RandomAgent.on_contract_nullified
      ~RandomAgent.on_contract_signed
      ~RandomAgent.on_contract_signed_
      ~RandomAgent.on_contracts_finalized
      ~RandomAgent.on_event
      ~RandomAgent.on_failures
      ~RandomAgent.on_neg_request_accepted
      ~RandomAgent.on_neg_request_accepted_
      ~RandomAgent.on_neg_request_rejected
      ~RandomAgent.on_neg_request_rejected_
      ~RandomAgent.on_negotiation_failure
      ~RandomAgent.on_negotiation_failure_
      ~RandomAgent.on_negotiation_success
      ~RandomAgent.on_negotiation_success_
      ~RandomAgent.read_config
      ~RandomAgent.respond_to_negotiation_request
      ~RandomAgent.respond_to_negotiation_request_
      ~RandomAgent.respond_to_renegotiation_request
      ~RandomAgent.set_renegotiation_agenda
      ~RandomAgent.sign_all_contracts
      ~RandomAgent.sign_contract
      ~RandomAgent.start_negotiations
      ~RandomAgent.step
      ~RandomAgent.step_

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

   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_exogenous_sales
   .. automethod:: confirm_exogenous_supplies
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: create_ufun
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: negotiator
   .. automethod:: notify
   .. automethod:: on_contract_breached
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_cancelled_
   .. automethod:: on_contract_executed
   .. automethod:: on_contract_nullified
   .. automethod:: on_contract_signed
   .. automethod:: on_contract_signed_
   .. automethod:: on_contracts_finalized
   .. automethod:: on_event
   .. automethod:: on_failures
   .. automethod:: on_neg_request_accepted
   .. automethod:: on_neg_request_accepted_
   .. automethod:: on_neg_request_rejected
   .. automethod:: on_neg_request_rejected_
   .. automethod:: on_negotiation_failure
   .. automethod:: on_negotiation_failure_
   .. automethod:: on_negotiation_success
   .. automethod:: on_negotiation_success_
   .. automethod:: read_config
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: start_negotiations
   .. automethod:: step
   .. automethod:: step_
