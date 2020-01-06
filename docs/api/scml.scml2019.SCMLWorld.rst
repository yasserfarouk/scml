SCMLWorld
=========

.. currentmodule:: scml.scml2019

.. autoclass:: SCMLWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCMLWorld.agreement_fraction
      ~SCMLWorld.agreement_rate
      ~SCMLWorld.breach_fraction
      ~SCMLWorld.breach_level
      ~SCMLWorld.breach_rate
      ~SCMLWorld.business_size
      ~SCMLWorld.cancellation_fraction
      ~SCMLWorld.cancellation_rate
      ~SCMLWorld.cancelled_contracts
      ~SCMLWorld.contract_dropping_fraction
      ~SCMLWorld.contract_err_fraction
      ~SCMLWorld.contract_execution_fraction
      ~SCMLWorld.contract_nullification_fraction
      ~SCMLWorld.current_step
      ~SCMLWorld.erred_contracts
      ~SCMLWorld.executed_contracts
      ~SCMLWorld.id
      ~SCMLWorld.log_folder
      ~SCMLWorld.n_negotiation_rounds_failed
      ~SCMLWorld.n_negotiation_rounds_successful
      ~SCMLWorld.name
      ~SCMLWorld.nullified_contracts
      ~SCMLWorld.relative_time
      ~SCMLWorld.remaining_steps
      ~SCMLWorld.remaining_time
      ~SCMLWorld.resolved_breaches
      ~SCMLWorld.saved_breaches
      ~SCMLWorld.saved_contracts
      ~SCMLWorld.saved_negotiations
      ~SCMLWorld.signed_contracts
      ~SCMLWorld.stats
      ~SCMLWorld.time
      ~SCMLWorld.unresolved_breaches
      ~SCMLWorld.uuid
      ~SCMLWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCMLWorld.announce
      ~SCMLWorld.append_stats
      ~SCMLWorld.assign_managers
      ~SCMLWorld.breach_record
      ~SCMLWorld.buy_insurance
      ~SCMLWorld.chain_world
      ~SCMLWorld.checkpoint
      ~SCMLWorld.checkpoint_final_step
      ~SCMLWorld.checkpoint_info
      ~SCMLWorld.checkpoint_init
      ~SCMLWorld.checkpoint_on_step_started
      ~SCMLWorld.complete_contract_execution
      ~SCMLWorld.contract_record
      ~SCMLWorld.contract_size
      ~SCMLWorld.create
      ~SCMLWorld.delete_executed_contracts
      ~SCMLWorld.evaluate_insurance
      ~SCMLWorld.executable_contracts
      ~SCMLWorld.execute_action
      ~SCMLWorld.from_checkpoint
      ~SCMLWorld.from_config
      ~SCMLWorld.get_dropped_contracts
      ~SCMLWorld.get_private_state
      ~SCMLWorld.ignore_contract
      ~SCMLWorld.init
      ~SCMLWorld.join
      ~SCMLWorld.logdebug
      ~SCMLWorld.logdebug_agent
      ~SCMLWorld.logerror
      ~SCMLWorld.logerror_agent
      ~SCMLWorld.loginfo
      ~SCMLWorld.loginfo_agent
      ~SCMLWorld.logwarning
      ~SCMLWorld.logwarning_agent
      ~SCMLWorld.make_bankrupt
      ~SCMLWorld.n_saved_contracts
      ~SCMLWorld.nullify_contract
      ~SCMLWorld.on_contract_cancelled
      ~SCMLWorld.on_contract_concluded
      ~SCMLWorld.on_contract_processed
      ~SCMLWorld.on_contract_signed
      ~SCMLWorld.on_event
      ~SCMLWorld.order_contracts_for_execution
      ~SCMLWorld.post_step_stats
      ~SCMLWorld.pre_step_stats
      ~SCMLWorld.random
      ~SCMLWorld.random_small
      ~SCMLWorld.read_config
      ~SCMLWorld.receive_financial_reports
      ~SCMLWorld.register
      ~SCMLWorld.register_interest
      ~SCMLWorld.register_listener
      ~SCMLWorld.register_stats_monitor
      ~SCMLWorld.register_world_monitor
      ~SCMLWorld.request_negotiation_about
      ~SCMLWorld.run
      ~SCMLWorld.run_negotiation
      ~SCMLWorld.run_negotiations
      ~SCMLWorld.save_config
      ~SCMLWorld.set_bulletin_board
      ~SCMLWorld.set_consumers
      ~SCMLWorld.set_factory_managers
      ~SCMLWorld.set_miners
      ~SCMLWorld.set_processes
      ~SCMLWorld.set_products
      ~SCMLWorld.simulation_step_after_execution
      ~SCMLWorld.simulation_step_before_execution
      ~SCMLWorld.start_contract_execution
      ~SCMLWorld.step
      ~SCMLWorld.unregister_interest
      ~SCMLWorld.unregister_stats_monitor
      ~SCMLWorld.unregister_world_monitor

   .. rubric:: Attributes Documentation

   .. autoattribute:: agreement_fraction
   .. autoattribute:: agreement_rate
   .. autoattribute:: breach_fraction
   .. autoattribute:: breach_level
   .. autoattribute:: breach_rate
   .. autoattribute:: business_size
   .. autoattribute:: cancellation_fraction
   .. autoattribute:: cancellation_rate
   .. autoattribute:: cancelled_contracts
   .. autoattribute:: contract_dropping_fraction
   .. autoattribute:: contract_err_fraction
   .. autoattribute:: contract_execution_fraction
   .. autoattribute:: contract_nullification_fraction
   .. autoattribute:: current_step
   .. autoattribute:: erred_contracts
   .. autoattribute:: executed_contracts
   .. autoattribute:: id
   .. autoattribute:: log_folder
   .. autoattribute:: n_negotiation_rounds_failed
   .. autoattribute:: n_negotiation_rounds_successful
   .. autoattribute:: name
   .. autoattribute:: nullified_contracts
   .. autoattribute:: relative_time
   .. autoattribute:: remaining_steps
   .. autoattribute:: remaining_time
   .. autoattribute:: resolved_breaches
   .. autoattribute:: saved_breaches
   .. autoattribute:: saved_contracts
   .. autoattribute:: saved_negotiations
   .. autoattribute:: signed_contracts
   .. autoattribute:: stats
   .. autoattribute:: time
   .. autoattribute:: unresolved_breaches
   .. autoattribute:: uuid
   .. autoattribute:: winners

   .. rubric:: Methods Documentation

   .. automethod:: announce
   .. automethod:: append_stats
   .. automethod:: assign_managers
   .. automethod:: breach_record
   .. automethod:: buy_insurance
   .. automethod:: chain_world
   .. automethod:: checkpoint
   .. automethod:: checkpoint_final_step
   .. automethod:: checkpoint_info
   .. automethod:: checkpoint_init
   .. automethod:: checkpoint_on_step_started
   .. automethod:: complete_contract_execution
   .. automethod:: contract_record
   .. automethod:: contract_size
   .. automethod:: create
   .. automethod:: delete_executed_contracts
   .. automethod:: evaluate_insurance
   .. automethod:: executable_contracts
   .. automethod:: execute_action
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: get_dropped_contracts
   .. automethod:: get_private_state
   .. automethod:: ignore_contract
   .. automethod:: init
   .. automethod:: join
   .. automethod:: logdebug
   .. automethod:: logdebug_agent
   .. automethod:: logerror
   .. automethod:: logerror_agent
   .. automethod:: loginfo
   .. automethod:: loginfo_agent
   .. automethod:: logwarning
   .. automethod:: logwarning_agent
   .. automethod:: make_bankrupt
   .. automethod:: n_saved_contracts
   .. automethod:: nullify_contract
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_concluded
   .. automethod:: on_contract_processed
   .. automethod:: on_contract_signed
   .. automethod:: on_event
   .. automethod:: order_contracts_for_execution
   .. automethod:: post_step_stats
   .. automethod:: pre_step_stats
   .. automethod:: random
   .. automethod:: random_small
   .. automethod:: read_config
   .. automethod:: receive_financial_reports
   .. automethod:: register
   .. automethod:: register_interest
   .. automethod:: register_listener
   .. automethod:: register_stats_monitor
   .. automethod:: register_world_monitor
   .. automethod:: request_negotiation_about
   .. automethod:: run
   .. automethod:: run_negotiation
   .. automethod:: run_negotiations
   .. automethod:: save_config
   .. automethod:: set_bulletin_board
   .. automethod:: set_consumers
   .. automethod:: set_factory_managers
   .. automethod:: set_miners
   .. automethod:: set_processes
   .. automethod:: set_products
   .. automethod:: simulation_step_after_execution
   .. automethod:: simulation_step_before_execution
   .. automethod:: start_contract_execution
   .. automethod:: step
   .. automethod:: unregister_interest
   .. automethod:: unregister_stats_monitor
   .. automethod:: unregister_world_monitor
