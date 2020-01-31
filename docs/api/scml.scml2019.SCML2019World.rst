SCML2019World
=============

.. currentmodule:: scml.scml2019

.. autoclass:: SCML2019World
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2019World.agreement_fraction
      ~SCML2019World.agreement_rate
      ~SCML2019World.breach_fraction
      ~SCML2019World.breach_level
      ~SCML2019World.breach_rate
      ~SCML2019World.business_size
      ~SCML2019World.cancellation_fraction
      ~SCML2019World.cancellation_rate
      ~SCML2019World.cancelled_contracts
      ~SCML2019World.contract_dropping_fraction
      ~SCML2019World.contract_err_fraction
      ~SCML2019World.contract_execution_fraction
      ~SCML2019World.contract_nullification_fraction
      ~SCML2019World.current_step
      ~SCML2019World.erred_contracts
      ~SCML2019World.executed_contracts
      ~SCML2019World.id
      ~SCML2019World.log_folder
      ~SCML2019World.n_negotiation_rounds_failed
      ~SCML2019World.n_negotiation_rounds_successful
      ~SCML2019World.name
      ~SCML2019World.nullified_contracts
      ~SCML2019World.relative_time
      ~SCML2019World.remaining_steps
      ~SCML2019World.remaining_time
      ~SCML2019World.resolved_breaches
      ~SCML2019World.saved_breaches
      ~SCML2019World.saved_contracts
      ~SCML2019World.saved_negotiations
      ~SCML2019World.signed_contracts
      ~SCML2019World.stats
      ~SCML2019World.time
      ~SCML2019World.total_time
      ~SCML2019World.unresolved_breaches
      ~SCML2019World.uuid
      ~SCML2019World.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2019World.announce
      ~SCML2019World.append_stats
      ~SCML2019World.assign_managers
      ~SCML2019World.breach_record
      ~SCML2019World.buy_insurance
      ~SCML2019World.chain_world
      ~SCML2019World.checkpoint
      ~SCML2019World.checkpoint_final_step
      ~SCML2019World.checkpoint_info
      ~SCML2019World.checkpoint_init
      ~SCML2019World.checkpoint_on_step_started
      ~SCML2019World.complete_contract_execution
      ~SCML2019World.contract_record
      ~SCML2019World.contract_size
      ~SCML2019World.create
      ~SCML2019World.delete_executed_contracts
      ~SCML2019World.draw
      ~SCML2019World.evaluate_insurance
      ~SCML2019World.executable_contracts
      ~SCML2019World.execute_action
      ~SCML2019World.from_checkpoint
      ~SCML2019World.from_config
      ~SCML2019World.get_dropped_contracts
      ~SCML2019World.get_private_state
      ~SCML2019World.graph
      ~SCML2019World.ignore_contract
      ~SCML2019World.init
      ~SCML2019World.join
      ~SCML2019World.logdebug
      ~SCML2019World.logdebug_agent
      ~SCML2019World.logerror
      ~SCML2019World.logerror_agent
      ~SCML2019World.loginfo
      ~SCML2019World.loginfo_agent
      ~SCML2019World.logwarning
      ~SCML2019World.logwarning_agent
      ~SCML2019World.make_bankrupt
      ~SCML2019World.n_saved_contracts
      ~SCML2019World.nullify_contract
      ~SCML2019World.on_contract_cancelled
      ~SCML2019World.on_contract_concluded
      ~SCML2019World.on_contract_processed
      ~SCML2019World.on_contract_signed
      ~SCML2019World.on_event
      ~SCML2019World.order_contracts_for_execution
      ~SCML2019World.post_step_stats
      ~SCML2019World.pre_step_stats
      ~SCML2019World.random
      ~SCML2019World.random_small
      ~SCML2019World.read_config
      ~SCML2019World.receive_financial_reports
      ~SCML2019World.register
      ~SCML2019World.register_interest
      ~SCML2019World.register_listener
      ~SCML2019World.register_stats_monitor
      ~SCML2019World.register_world_monitor
      ~SCML2019World.request_negotiation_about
      ~SCML2019World.run
      ~SCML2019World.run_negotiation
      ~SCML2019World.run_negotiations
      ~SCML2019World.save_config
      ~SCML2019World.set_bulletin_board
      ~SCML2019World.set_consumers
      ~SCML2019World.set_factory_managers
      ~SCML2019World.set_miners
      ~SCML2019World.set_processes
      ~SCML2019World.set_products
      ~SCML2019World.simulation_step
      ~SCML2019World.start_contract_execution
      ~SCML2019World.step
      ~SCML2019World.unregister_interest
      ~SCML2019World.unregister_stats_monitor
      ~SCML2019World.unregister_world_monitor
      ~SCML2019World.update_stats

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
   .. autoattribute:: total_time
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
   .. automethod:: draw
   .. automethod:: evaluate_insurance
   .. automethod:: executable_contracts
   .. automethod:: execute_action
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: get_dropped_contracts
   .. automethod:: get_private_state
   .. automethod:: graph
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
   .. automethod:: simulation_step
   .. automethod:: start_contract_execution
   .. automethod:: step
   .. automethod:: unregister_interest
   .. automethod:: unregister_stats_monitor
   .. automethod:: unregister_world_monitor
   .. automethod:: update_stats
