DecentralizingAgent
===================

.. currentmodule:: scml.scml2020

.. autoclass:: DecentralizingAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DecentralizingAgent.accepted_negotiation_requests
      ~DecentralizingAgent.awi
      ~DecentralizingAgent.id
      ~DecentralizingAgent.initialized
      ~DecentralizingAgent.name
      ~DecentralizingAgent.negotiation_requests
      ~DecentralizingAgent.requested_negotiations
      ~DecentralizingAgent.running_negotiations
      ~DecentralizingAgent.short_type_name
      ~DecentralizingAgent.type_name
      ~DecentralizingAgent.unsigned_contracts
      ~DecentralizingAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~DecentralizingAgent.add_controller
      ~DecentralizingAgent.all_negotiations_concluded
      ~DecentralizingAgent.checkpoint
      ~DecentralizingAgent.checkpoint_info
      ~DecentralizingAgent.confirm_exogenous_sales
      ~DecentralizingAgent.confirm_exogenous_supplies
      ~DecentralizingAgent.confirm_production
      ~DecentralizingAgent.create
      ~DecentralizingAgent.create_negotiation_request
      ~DecentralizingAgent.from_checkpoint
      ~DecentralizingAgent.from_config
      ~DecentralizingAgent.generate_buy_negotiations
      ~DecentralizingAgent.generate_sell_negotiations
      ~DecentralizingAgent.init
      ~DecentralizingAgent.init_
      ~DecentralizingAgent.max_consumption_till
      ~DecentralizingAgent.max_production_till
      ~DecentralizingAgent.notify
      ~DecentralizingAgent.on_contract_breached
      ~DecentralizingAgent.on_contract_cancelled
      ~DecentralizingAgent.on_contract_cancelled_
      ~DecentralizingAgent.on_contract_executed
      ~DecentralizingAgent.on_contract_nullified
      ~DecentralizingAgent.on_contract_signed
      ~DecentralizingAgent.on_contract_signed_
      ~DecentralizingAgent.on_contracts_finalized
      ~DecentralizingAgent.on_event
      ~DecentralizingAgent.on_failures
      ~DecentralizingAgent.on_neg_request_accepted
      ~DecentralizingAgent.on_neg_request_accepted_
      ~DecentralizingAgent.on_neg_request_rejected
      ~DecentralizingAgent.on_neg_request_rejected_
      ~DecentralizingAgent.on_negotiation_failure
      ~DecentralizingAgent.on_negotiation_failure_
      ~DecentralizingAgent.on_negotiation_success
      ~DecentralizingAgent.on_negotiation_success_
      ~DecentralizingAgent.read_config
      ~DecentralizingAgent.respond_to_negotiation_request
      ~DecentralizingAgent.respond_to_negotiation_request_
      ~DecentralizingAgent.respond_to_renegotiation_request
      ~DecentralizingAgent.set_renegotiation_agenda
      ~DecentralizingAgent.sign_all_contracts
      ~DecentralizingAgent.sign_contract
      ~DecentralizingAgent.start_negotiations
      ~DecentralizingAgent.step
      ~DecentralizingAgent.step_

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

   .. automethod:: add_controller
   .. automethod:: all_negotiations_concluded
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_exogenous_sales
   .. automethod:: confirm_exogenous_supplies
   .. automethod:: confirm_production
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: generate_buy_negotiations
   .. automethod:: generate_sell_negotiations
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: max_consumption_till
   .. automethod:: max_production_till
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
