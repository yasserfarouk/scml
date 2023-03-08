JustInTimeConsumer
==================

.. currentmodule:: scml.scml2019

.. autoclass:: JustInTimeConsumer
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~JustInTimeConsumer.MAX_UNIT_PRICE
      ~JustInTimeConsumer.RELATIVE_MAX_PRICE
      ~JustInTimeConsumer.accepted_negotiation_requests
      ~JustInTimeConsumer.awi
      ~JustInTimeConsumer.crisp_ufun
      ~JustInTimeConsumer.has_cardinal_preferences
      ~JustInTimeConsumer.has_preferences
      ~JustInTimeConsumer.has_ufun
      ~JustInTimeConsumer.id
      ~JustInTimeConsumer.initialized
      ~JustInTimeConsumer.name
      ~JustInTimeConsumer.negotiation_requests
      ~JustInTimeConsumer.preferences
      ~JustInTimeConsumer.prob_ufun
      ~JustInTimeConsumer.requested_negotiations
      ~JustInTimeConsumer.reserved_outcome
      ~JustInTimeConsumer.reserved_value
      ~JustInTimeConsumer.running_negotiations
      ~JustInTimeConsumer.short_type_name
      ~JustInTimeConsumer.type_name
      ~JustInTimeConsumer.type_postfix
      ~JustInTimeConsumer.ufun
      ~JustInTimeConsumer.unsigned_contracts
      ~JustInTimeConsumer.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~JustInTimeConsumer.can_expect_agreement
      ~JustInTimeConsumer.checkpoint
      ~JustInTimeConsumer.checkpoint_info
      ~JustInTimeConsumer.confirm_contract_execution
      ~JustInTimeConsumer.confirm_loan
      ~JustInTimeConsumer.confirm_partial_execution
      ~JustInTimeConsumer.create
      ~JustInTimeConsumer.create_negotiation_request
      ~JustInTimeConsumer.from_checkpoint
      ~JustInTimeConsumer.from_config
      ~JustInTimeConsumer.init
      ~JustInTimeConsumer.init_
      ~JustInTimeConsumer.notify
      ~JustInTimeConsumer.on_agent_bankrupt
      ~JustInTimeConsumer.on_cash_transfer
      ~JustInTimeConsumer.on_contract_breached
      ~JustInTimeConsumer.on_contract_cancelled
      ~JustInTimeConsumer.on_contract_cancelled_
      ~JustInTimeConsumer.on_contract_executed
      ~JustInTimeConsumer.on_contract_nullified
      ~JustInTimeConsumer.on_contract_signed
      ~JustInTimeConsumer.on_contract_signed_
      ~JustInTimeConsumer.on_contracts_finalized
      ~JustInTimeConsumer.on_event
      ~JustInTimeConsumer.on_inventory_change
      ~JustInTimeConsumer.on_neg_request_accepted
      ~JustInTimeConsumer.on_neg_request_accepted_
      ~JustInTimeConsumer.on_neg_request_rejected
      ~JustInTimeConsumer.on_neg_request_rejected_
      ~JustInTimeConsumer.on_negotiation_failure
      ~JustInTimeConsumer.on_negotiation_failure_
      ~JustInTimeConsumer.on_negotiation_success
      ~JustInTimeConsumer.on_negotiation_success_
      ~JustInTimeConsumer.on_new_cfp
      ~JustInTimeConsumer.on_new_report
      ~JustInTimeConsumer.on_preferences_changed
      ~JustInTimeConsumer.on_remove_cfp
      ~JustInTimeConsumer.on_simulation_step_ended
      ~JustInTimeConsumer.on_simulation_step_started
      ~JustInTimeConsumer.read_config
      ~JustInTimeConsumer.register_product_cfps
      ~JustInTimeConsumer.request_negotiation
      ~JustInTimeConsumer.respond_to_negotiation_request
      ~JustInTimeConsumer.respond_to_negotiation_request_
      ~JustInTimeConsumer.respond_to_renegotiation_request
      ~JustInTimeConsumer.set_preferences
      ~JustInTimeConsumer.set_profiles
      ~JustInTimeConsumer.set_renegotiation_agenda
      ~JustInTimeConsumer.sign_all_contracts
      ~JustInTimeConsumer.sign_contract
      ~JustInTimeConsumer.spawn
      ~JustInTimeConsumer.spawn_object
      ~JustInTimeConsumer.step
      ~JustInTimeConsumer.step_

   .. rubric:: Attributes Documentation

   .. autoattribute:: MAX_UNIT_PRICE
   .. autoattribute:: RELATIVE_MAX_PRICE
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
   .. automethod:: on_preferences_changed
   .. automethod:: on_remove_cfp
   .. automethod:: on_simulation_step_ended
   .. automethod:: on_simulation_step_started
   .. automethod:: read_config
   .. automethod:: register_product_cfps
   .. automethod:: request_negotiation
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: set_preferences
   .. automethod:: set_profiles
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: step
   .. automethod:: step_
