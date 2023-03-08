Bank
====

.. currentmodule:: scml.scml2019

.. autoclass:: Bank
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Bank.accepted_negotiation_requests
      ~Bank.awi
      ~Bank.crisp_ufun
      ~Bank.has_cardinal_preferences
      ~Bank.has_preferences
      ~Bank.has_ufun
      ~Bank.id
      ~Bank.initialized
      ~Bank.name
      ~Bank.negotiation_requests
      ~Bank.preferences
      ~Bank.prob_ufun
      ~Bank.requested_negotiations
      ~Bank.reserved_outcome
      ~Bank.reserved_value
      ~Bank.running_negotiations
      ~Bank.short_type_name
      ~Bank.type_name
      ~Bank.type_postfix
      ~Bank.ufun
      ~Bank.unsigned_contracts
      ~Bank.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~Bank.checkpoint
      ~Bank.checkpoint_info
      ~Bank.create
      ~Bank.create_negotiation_request
      ~Bank.from_checkpoint
      ~Bank.from_config
      ~Bank.init
      ~Bank.init_
      ~Bank.notify
      ~Bank.on_contract_breached
      ~Bank.on_contract_cancelled
      ~Bank.on_contract_cancelled_
      ~Bank.on_contract_executed
      ~Bank.on_contract_signed
      ~Bank.on_contract_signed_
      ~Bank.on_contracts_finalized
      ~Bank.on_event
      ~Bank.on_neg_request_accepted
      ~Bank.on_neg_request_accepted_
      ~Bank.on_neg_request_rejected
      ~Bank.on_neg_request_rejected_
      ~Bank.on_negotiation_failure
      ~Bank.on_negotiation_failure_
      ~Bank.on_negotiation_success
      ~Bank.on_negotiation_success_
      ~Bank.on_preferences_changed
      ~Bank.on_simulation_step_ended
      ~Bank.on_simulation_step_started
      ~Bank.read_config
      ~Bank.respond_to_negotiation_request
      ~Bank.respond_to_negotiation_request_
      ~Bank.respond_to_renegotiation_request
      ~Bank.set_preferences
      ~Bank.set_renegotiation_agenda
      ~Bank.sign_all_contracts
      ~Bank.sign_contract
      ~Bank.spawn
      ~Bank.spawn_object
      ~Bank.step
      ~Bank.step_

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: has_ufun
   .. autoattribute:: id
   .. autoattribute:: initialized
   .. autoattribute:: name
   .. autoattribute:: negotiation_requests
   .. autoattribute:: preferences
   .. autoattribute:: prob_ufun
   .. autoattribute:: requested_negotiations
   .. autoattribute:: reserved_outcome
   .. autoattribute:: reserved_value
   .. autoattribute:: running_negotiations
   .. autoattribute:: short_type_name
   .. autoattribute:: type_name
   .. autoattribute:: type_postfix
   .. autoattribute:: ufun
   .. autoattribute:: unsigned_contracts
   .. autoattribute:: uuid

   .. rubric:: Methods Documentation

   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: notify
   .. automethod:: on_contract_breached
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_cancelled_
   .. automethod:: on_contract_executed
   .. automethod:: on_contract_signed
   .. automethod:: on_contract_signed_
   .. automethod:: on_contracts_finalized
   .. automethod:: on_event
   .. automethod:: on_neg_request_accepted
   .. automethod:: on_neg_request_accepted_
   .. automethod:: on_neg_request_rejected
   .. automethod:: on_neg_request_rejected_
   .. automethod:: on_negotiation_failure
   .. automethod:: on_negotiation_failure_
   .. automethod:: on_negotiation_success
   .. automethod:: on_negotiation_success_
   .. automethod:: on_preferences_changed
   .. automethod:: on_simulation_step_ended
   .. automethod:: on_simulation_step_started
   .. automethod:: read_config
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: set_preferences
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: step
   .. automethod:: step_
