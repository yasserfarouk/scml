ReactiveAgent
=============

.. currentmodule:: scml.scml2020

.. autoclass:: ReactiveAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~ReactiveAgent.accepted_negotiation_requests
      ~ReactiveAgent.awi
      ~ReactiveAgent.id
      ~ReactiveAgent.initialized
      ~ReactiveAgent.internal_state
      ~ReactiveAgent.name
      ~ReactiveAgent.negotiation_requests
      ~ReactiveAgent.requested_negotiations
      ~ReactiveAgent.running_negotiations
      ~ReactiveAgent.short_type_name
      ~ReactiveAgent.type_name
      ~ReactiveAgent.unsigned_contracts
      ~ReactiveAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~ReactiveAgent.acceptable_unit_price
      ~ReactiveAgent.add_controller
      ~ReactiveAgent.all_negotiations_concluded
      ~ReactiveAgent.can_be_produced
      ~ReactiveAgent.checkpoint
      ~ReactiveAgent.checkpoint_info
      ~ReactiveAgent.confirm_production
      ~ReactiveAgent.create
      ~ReactiveAgent.create_negotiation_request
      ~ReactiveAgent.from_checkpoint
      ~ReactiveAgent.from_config
      ~ReactiveAgent.init
      ~ReactiveAgent.init_
      ~ReactiveAgent.notify
      ~ReactiveAgent.on_agent_bankrupt
      ~ReactiveAgent.on_contract_breached
      ~ReactiveAgent.on_contract_cancelled
      ~ReactiveAgent.on_contract_cancelled_
      ~ReactiveAgent.on_contract_executed
      ~ReactiveAgent.on_contract_signed
      ~ReactiveAgent.on_contract_signed_
      ~ReactiveAgent.on_contracts_finalized
      ~ReactiveAgent.on_event
      ~ReactiveAgent.on_failures
      ~ReactiveAgent.on_neg_request_accepted
      ~ReactiveAgent.on_neg_request_accepted_
      ~ReactiveAgent.on_neg_request_rejected
      ~ReactiveAgent.on_neg_request_rejected_
      ~ReactiveAgent.on_negotiation_failure
      ~ReactiveAgent.on_negotiation_failure_
      ~ReactiveAgent.on_negotiation_success
      ~ReactiveAgent.on_negotiation_success_
      ~ReactiveAgent.on_simulation_step_ended
      ~ReactiveAgent.on_simulation_step_started
      ~ReactiveAgent.predict_quantity
      ~ReactiveAgent.read_config
      ~ReactiveAgent.respond_to_negotiation_request
      ~ReactiveAgent.respond_to_negotiation_request_
      ~ReactiveAgent.respond_to_renegotiation_request
      ~ReactiveAgent.set_renegotiation_agenda
      ~ReactiveAgent.sign_all_contracts
      ~ReactiveAgent.sign_contract
      ~ReactiveAgent.start_negotiations
      ~ReactiveAgent.step
      ~ReactiveAgent.step_
      ~ReactiveAgent.target_quantities
      ~ReactiveAgent.target_quantity

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: id
   .. autoattribute:: initialized
   .. autoattribute:: internal_state
   .. autoattribute:: name
   .. autoattribute:: negotiation_requests
   .. autoattribute:: requested_negotiations
   .. autoattribute:: running_negotiations
   .. autoattribute:: short_type_name
   .. autoattribute:: type_name
   .. autoattribute:: unsigned_contracts
   .. autoattribute:: uuid

   .. rubric:: Methods Documentation

   .. automethod:: acceptable_unit_price
   .. automethod:: add_controller
   .. automethod:: all_negotiations_concluded
   .. automethod:: can_be_produced
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: notify
   .. automethod:: on_agent_bankrupt
   .. automethod:: on_contract_breached
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_cancelled_
   .. automethod:: on_contract_executed
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
   .. automethod:: on_simulation_step_ended
   .. automethod:: on_simulation_step_started
   .. automethod:: predict_quantity
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
   .. automethod:: target_quantities
   .. automethod:: target_quantity
