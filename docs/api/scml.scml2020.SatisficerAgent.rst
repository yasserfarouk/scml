SatisficerAgent
===============

.. currentmodule:: scml.scml2020

.. autoclass:: SatisficerAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SatisficerAgent.accepted_negotiation_requests
      ~SatisficerAgent.awi
      ~SatisficerAgent.crisp_ufun
      ~SatisficerAgent.has_cardinal_preferences
      ~SatisficerAgent.has_preferences
      ~SatisficerAgent.id
      ~SatisficerAgent.initialized
      ~SatisficerAgent.internal_state
      ~SatisficerAgent.name
      ~SatisficerAgent.negotiation_requests
      ~SatisficerAgent.preferences
      ~SatisficerAgent.prob_ufun
      ~SatisficerAgent.requested_negotiations
      ~SatisficerAgent.reserved_outcome
      ~SatisficerAgent.reserved_value
      ~SatisficerAgent.running_negotiations
      ~SatisficerAgent.short_type_name
      ~SatisficerAgent.type_name
      ~SatisficerAgent.type_postfix
      ~SatisficerAgent.ufun
      ~SatisficerAgent.unsigned_contracts
      ~SatisficerAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~SatisficerAgent.before_step
      ~SatisficerAgent.checkpoint
      ~SatisficerAgent.checkpoint_info
      ~SatisficerAgent.confirm_production
      ~SatisficerAgent.create
      ~SatisficerAgent.create_negotiation_request
      ~SatisficerAgent.do_production
      ~SatisficerAgent.from_checkpoint
      ~SatisficerAgent.from_config
      ~SatisficerAgent.init
      ~SatisficerAgent.init_
      ~SatisficerAgent.notify
      ~SatisficerAgent.on_agent_bankrupt
      ~SatisficerAgent.on_contract_breached
      ~SatisficerAgent.on_contract_cancelled
      ~SatisficerAgent.on_contract_cancelled_
      ~SatisficerAgent.on_contract_executed
      ~SatisficerAgent.on_contract_signed
      ~SatisficerAgent.on_contract_signed_
      ~SatisficerAgent.on_contracts_finalized
      ~SatisficerAgent.on_event
      ~SatisficerAgent.on_failures
      ~SatisficerAgent.on_neg_request_accepted
      ~SatisficerAgent.on_neg_request_accepted_
      ~SatisficerAgent.on_neg_request_rejected
      ~SatisficerAgent.on_neg_request_rejected_
      ~SatisficerAgent.on_negotiation_failure
      ~SatisficerAgent.on_negotiation_failure_
      ~SatisficerAgent.on_negotiation_success
      ~SatisficerAgent.on_negotiation_success_
      ~SatisficerAgent.on_preferences_changed
      ~SatisficerAgent.on_simulation_step_ended
      ~SatisficerAgent.on_simulation_step_started
      ~SatisficerAgent.propose
      ~SatisficerAgent.read_config
      ~SatisficerAgent.respond
      ~SatisficerAgent.respond_to_negotiation_request
      ~SatisficerAgent.respond_to_negotiation_request_
      ~SatisficerAgent.respond_to_renegotiation_request
      ~SatisficerAgent.set_preferences
      ~SatisficerAgent.set_renegotiation_agenda
      ~SatisficerAgent.sign_all_contracts
      ~SatisficerAgent.sign_contract
      ~SatisficerAgent.spawn
      ~SatisficerAgent.spawn_object
      ~SatisficerAgent.step
      ~SatisficerAgent.step_
      ~SatisficerAgent.to_dict

   .. rubric:: Attributes Documentation

   .. autoattribute:: accepted_negotiation_requests
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: id
   .. autoattribute:: initialized
   .. autoattribute:: internal_state
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

   .. automethod:: before_step
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: do_production
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
   .. automethod:: on_preferences_changed
   .. automethod:: on_simulation_step_ended
   .. automethod:: on_simulation_step_started
   .. automethod:: propose
   .. automethod:: read_config
   .. automethod:: respond
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
   .. automethod:: to_dict
