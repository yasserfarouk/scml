CheapBuyerFactoryManager
========================

.. currentmodule:: scml.scml2019

.. autoclass:: CheapBuyerFactoryManager
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~CheapBuyerFactoryManager.accepted_negotiation_requests
      ~CheapBuyerFactoryManager.awi
      ~CheapBuyerFactoryManager.id
      ~CheapBuyerFactoryManager.initialized
      ~CheapBuyerFactoryManager.name
      ~CheapBuyerFactoryManager.negotiation_requests
      ~CheapBuyerFactoryManager.requested_negotiations
      ~CheapBuyerFactoryManager.running_negotiations
      ~CheapBuyerFactoryManager.short_type_name
      ~CheapBuyerFactoryManager.type_name
      ~CheapBuyerFactoryManager.unsigned_contracts
      ~CheapBuyerFactoryManager.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~CheapBuyerFactoryManager.accept_negotiation
      ~CheapBuyerFactoryManager.accept_negotiation_old
      ~CheapBuyerFactoryManager.add_negotiation_step_data
      ~CheapBuyerFactoryManager.add_to_cumulative_estimated_demands
      ~CheapBuyerFactoryManager.can_expect_agreement
      ~CheapBuyerFactoryManager.can_produce
      ~CheapBuyerFactoryManager.can_secure_needs
      ~CheapBuyerFactoryManager.checkpoint
      ~CheapBuyerFactoryManager.checkpoint_info
      ~CheapBuyerFactoryManager.confirm_contract_execution
      ~CheapBuyerFactoryManager.confirm_loan
      ~CheapBuyerFactoryManager.confirm_partial_execution
      ~CheapBuyerFactoryManager.create
      ~CheapBuyerFactoryManager.create_negotiation_request
      ~CheapBuyerFactoryManager.estimate_unweighted_demand
      ~CheapBuyerFactoryManager.estimate_weighted_demand
      ~CheapBuyerFactoryManager.from_checkpoint
      ~CheapBuyerFactoryManager.from_config
      ~CheapBuyerFactoryManager.generateDecoyCFPs
      ~CheapBuyerFactoryManager.get_alpha_and_beta
      ~CheapBuyerFactoryManager.get_amount_of_final_products
      ~CheapBuyerFactoryManager.get_amount_of_raw_materials
      ~CheapBuyerFactoryManager.get_average_buying_price
      ~CheapBuyerFactoryManager.get_average_selling_price
      ~CheapBuyerFactoryManager.get_balance
      ~CheapBuyerFactoryManager.get_process_cost
      ~CheapBuyerFactoryManager.get_sum_of_unweighted_demands
      ~CheapBuyerFactoryManager.get_sum_of_weighted_demands
      ~CheapBuyerFactoryManager.get_target_price
      ~CheapBuyerFactoryManager.get_total_profit
      ~CheapBuyerFactoryManager.get_unweighted_estimated_average_demands
      ~CheapBuyerFactoryManager.init
      ~CheapBuyerFactoryManager.init_
      ~CheapBuyerFactoryManager.initialize_unweighted_sum_of_estimated_demands
      ~CheapBuyerFactoryManager.initialize_weighted_sum_of_estimated_demands
      ~CheapBuyerFactoryManager.is_cfp_acceptable
      ~CheapBuyerFactoryManager.is_cfp_acceptable_2
      ~CheapBuyerFactoryManager.notify
      ~CheapBuyerFactoryManager.on_agent_bankrupt
      ~CheapBuyerFactoryManager.on_cash_transfer
      ~CheapBuyerFactoryManager.on_contract_breached
      ~CheapBuyerFactoryManager.on_contract_cancelled
      ~CheapBuyerFactoryManager.on_contract_cancelled_
      ~CheapBuyerFactoryManager.on_contract_executed
      ~CheapBuyerFactoryManager.on_contract_nullified
      ~CheapBuyerFactoryManager.on_contract_signed
      ~CheapBuyerFactoryManager.on_contract_signed_
      ~CheapBuyerFactoryManager.on_contracts_finalized
      ~CheapBuyerFactoryManager.on_event
      ~CheapBuyerFactoryManager.on_inventory_change
      ~CheapBuyerFactoryManager.on_neg_request_accepted
      ~CheapBuyerFactoryManager.on_neg_request_accepted_
      ~CheapBuyerFactoryManager.on_neg_request_rejected
      ~CheapBuyerFactoryManager.on_neg_request_rejected_
      ~CheapBuyerFactoryManager.on_negotiation_failure
      ~CheapBuyerFactoryManager.on_negotiation_failure_
      ~CheapBuyerFactoryManager.on_negotiation_success
      ~CheapBuyerFactoryManager.on_negotiation_success_
      ~CheapBuyerFactoryManager.on_new_cfp
      ~CheapBuyerFactoryManager.on_new_report
      ~CheapBuyerFactoryManager.on_production_failure
      ~CheapBuyerFactoryManager.on_production_success
      ~CheapBuyerFactoryManager.on_remove_cfp
      ~CheapBuyerFactoryManager.post_cfps
      ~CheapBuyerFactoryManager.post_cfps_2
      ~CheapBuyerFactoryManager.process_raw_materials
      ~CheapBuyerFactoryManager.read_config
      ~CheapBuyerFactoryManager.register_all_products
      ~CheapBuyerFactoryManager.request_negotiation
      ~CheapBuyerFactoryManager.respond_to_cfp
      ~CheapBuyerFactoryManager.respond_to_negotiation_request
      ~CheapBuyerFactoryManager.respond_to_negotiation_request_
      ~CheapBuyerFactoryManager.respond_to_renegotiation_request
      ~CheapBuyerFactoryManager.set_renegotiation_agenda
      ~CheapBuyerFactoryManager.sign_all_contracts
      ~CheapBuyerFactoryManager.sign_contract
      ~CheapBuyerFactoryManager.step
      ~CheapBuyerFactoryManager.step_
      ~CheapBuyerFactoryManager.total_utility
      ~CheapBuyerFactoryManager.unweighted_sum_of_estimated_demands
      ~CheapBuyerFactoryManager.write_to_file

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

   .. automethod:: accept_negotiation
   .. automethod:: accept_negotiation_old
   .. automethod:: add_negotiation_step_data
   .. automethod:: add_to_cumulative_estimated_demands
   .. automethod:: can_expect_agreement
   .. automethod:: can_produce
   .. automethod:: can_secure_needs
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: confirm_contract_execution
   .. automethod:: confirm_loan
   .. automethod:: confirm_partial_execution
   .. automethod:: create
   .. automethod:: create_negotiation_request
   .. automethod:: estimate_unweighted_demand
   .. automethod:: estimate_weighted_demand
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: generateDecoyCFPs
   .. automethod:: get_alpha_and_beta
   .. automethod:: get_amount_of_final_products
   .. automethod:: get_amount_of_raw_materials
   .. automethod:: get_average_buying_price
   .. automethod:: get_average_selling_price
   .. automethod:: get_balance
   .. automethod:: get_process_cost
   .. automethod:: get_sum_of_unweighted_demands
   .. automethod:: get_sum_of_weighted_demands
   .. automethod:: get_target_price
   .. automethod:: get_total_profit
   .. automethod:: get_unweighted_estimated_average_demands
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: initialize_unweighted_sum_of_estimated_demands
   .. automethod:: initialize_weighted_sum_of_estimated_demands
   .. automethod:: is_cfp_acceptable
   .. automethod:: is_cfp_acceptable_2
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
   .. automethod:: on_production_failure
   .. automethod:: on_production_success
   .. automethod:: on_remove_cfp
   .. automethod:: post_cfps
   .. automethod:: post_cfps_2
   .. automethod:: process_raw_materials
   .. automethod:: read_config
   .. automethod:: register_all_products
   .. automethod:: request_negotiation
   .. automethod:: respond_to_cfp
   .. automethod:: respond_to_negotiation_request
   .. automethod:: respond_to_negotiation_request_
   .. automethod:: respond_to_renegotiation_request
   .. automethod:: set_renegotiation_agenda
   .. automethod:: sign_all_contracts
   .. automethod:: sign_contract
   .. automethod:: step
   .. automethod:: step_
   .. automethod:: total_utility
   .. automethod:: unweighted_sum_of_estimated_demands
   .. automethod:: write_to_file
