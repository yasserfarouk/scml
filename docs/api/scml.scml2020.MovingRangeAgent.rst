MovingRangeAgent
================

.. currentmodule:: scml.scml2020

.. autoclass:: MovingRangeAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MovingRangeAgent.accepted_negotiation_requests
      ~MovingRangeAgent.awi
      ~MovingRangeAgent.id
      ~MovingRangeAgent.initialized
      ~MovingRangeAgent.internal_state
      ~MovingRangeAgent.name
      ~MovingRangeAgent.negotiation_requests
      ~MovingRangeAgent.requested_negotiations
      ~MovingRangeAgent.running_negotiations
      ~MovingRangeAgent.short_type_name
      ~MovingRangeAgent.type_name
      ~MovingRangeAgent.unsigned_contracts
      ~MovingRangeAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~MovingRangeAgent.can_be_produced
      ~MovingRangeAgent.checkpoint
      ~MovingRangeAgent.checkpoint_info
      ~MovingRangeAgent.confirm_production
      ~MovingRangeAgent.create
      ~MovingRangeAgent.create_negotiation_request
      ~MovingRangeAgent.from_checkpoint
      ~MovingRangeAgent.from_config
      ~MovingRangeAgent.init
      ~MovingRangeAgent.init_
      ~MovingRangeAgent.notify
      ~MovingRangeAgent.on_agent_bankrupt
      ~MovingRangeAgent.on_contract_breached
      ~MovingRangeAgent.on_contract_cancelled
      ~MovingRangeAgent.on_contract_cancelled_
      ~MovingRangeAgent.on_contract_executed
      ~MovingRangeAgent.on_contract_signed
      ~MovingRangeAgent.on_contract_signed_
      ~MovingRangeAgent.on_contracts_finalized
      ~MovingRangeAgent.on_event
      ~MovingRangeAgent.on_failures
      ~MovingRangeAgent.on_neg_request_accepted
      ~MovingRangeAgent.on_neg_request_accepted_
      ~MovingRangeAgent.on_neg_request_rejected
      ~MovingRangeAgent.on_neg_request_rejected_
      ~MovingRangeAgent.on_negotiation_failure
      ~MovingRangeAgent.on_negotiation_failure_
      ~MovingRangeAgent.on_negotiation_success
      ~MovingRangeAgent.on_negotiation_success_
      ~MovingRangeAgent.on_simulation_step_ended
      ~MovingRangeAgent.on_simulation_step_started
      ~MovingRangeAgent.predict_quantity
      ~MovingRangeAgent.read_config
      ~MovingRangeAgent.respond_to_negotiation_request
      ~MovingRangeAgent.respond_to_negotiation_request_
      ~MovingRangeAgent.respond_to_renegotiation_request
      ~MovingRangeAgent.set_renegotiation_agenda
      ~MovingRangeAgent.sign_all_contracts
      ~MovingRangeAgent.sign_contract
      ~MovingRangeAgent.step
      ~MovingRangeAgent.step_

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
   .. automethod:: step
   .. automethod:: step_
