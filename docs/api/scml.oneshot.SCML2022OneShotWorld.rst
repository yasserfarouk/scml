SCML2022OneShotWorld
====================

.. currentmodule:: scml.oneshot

.. autoclass:: SCML2022OneShotWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2022OneShotWorld.agreement_fraction
      ~SCML2022OneShotWorld.agreement_rate
      ~SCML2022OneShotWorld.breach_fraction
      ~SCML2022OneShotWorld.breach_level
      ~SCML2022OneShotWorld.breach_rate
      ~SCML2022OneShotWorld.business_size
      ~SCML2022OneShotWorld.cancellation_fraction
      ~SCML2022OneShotWorld.cancellation_rate
      ~SCML2022OneShotWorld.cancelled_contracts
      ~SCML2022OneShotWorld.contract_dropping_fraction
      ~SCML2022OneShotWorld.contract_err_fraction
      ~SCML2022OneShotWorld.contract_execution_fraction
      ~SCML2022OneShotWorld.contract_nullification_fraction
      ~SCML2022OneShotWorld.contracts_df
      ~SCML2022OneShotWorld.current_step
      ~SCML2022OneShotWorld.erred_contracts
      ~SCML2022OneShotWorld.executed_contracts
      ~SCML2022OneShotWorld.id
      ~SCML2022OneShotWorld.log_folder
      ~SCML2022OneShotWorld.n_agent_exceptions
      ~SCML2022OneShotWorld.n_contract_exceptions
      ~SCML2022OneShotWorld.n_mechanism_exceptions
      ~SCML2022OneShotWorld.n_negotiation_rounds_failed
      ~SCML2022OneShotWorld.n_negotiation_rounds_successful
      ~SCML2022OneShotWorld.n_negotiator_exceptions
      ~SCML2022OneShotWorld.n_simulation_exceptions
      ~SCML2022OneShotWorld.n_total_agent_exceptions
      ~SCML2022OneShotWorld.n_total_simulation_exceptions
      ~SCML2022OneShotWorld.name
      ~SCML2022OneShotWorld.non_system_agent_ids
      ~SCML2022OneShotWorld.non_system_agent_names
      ~SCML2022OneShotWorld.non_system_agents
      ~SCML2022OneShotWorld.nullified_contracts
      ~SCML2022OneShotWorld.relative_time
      ~SCML2022OneShotWorld.remaining_steps
      ~SCML2022OneShotWorld.remaining_time
      ~SCML2022OneShotWorld.resolved_breaches
      ~SCML2022OneShotWorld.saved_breaches
      ~SCML2022OneShotWorld.saved_contracts
      ~SCML2022OneShotWorld.saved_negotiations
      ~SCML2022OneShotWorld.short_type_name
      ~SCML2022OneShotWorld.signed_contracts
      ~SCML2022OneShotWorld.stats
      ~SCML2022OneShotWorld.stats_df
      ~SCML2022OneShotWorld.system_agent_ids
      ~SCML2022OneShotWorld.system_agent_names
      ~SCML2022OneShotWorld.system_agents
      ~SCML2022OneShotWorld.time
      ~SCML2022OneShotWorld.total_time
      ~SCML2022OneShotWorld.trading_prices
      ~SCML2022OneShotWorld.type_name
      ~SCML2022OneShotWorld.unresolved_breaches
      ~SCML2022OneShotWorld.uuid
      ~SCML2022OneShotWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2022OneShotWorld.add_financial_report
      ~SCML2022OneShotWorld.announce
      ~SCML2022OneShotWorld.append_stats
      ~SCML2022OneShotWorld.breach_record
      ~SCML2022OneShotWorld.call
      ~SCML2022OneShotWorld.checkpoint
      ~SCML2022OneShotWorld.checkpoint_final_step
      ~SCML2022OneShotWorld.checkpoint_info
      ~SCML2022OneShotWorld.checkpoint_init
      ~SCML2022OneShotWorld.checkpoint_on_step_started
      ~SCML2022OneShotWorld.complete_contract_execution
      ~SCML2022OneShotWorld.contract_record
      ~SCML2022OneShotWorld.contract_size
      ~SCML2022OneShotWorld.create
      ~SCML2022OneShotWorld.current_balance
      ~SCML2022OneShotWorld.delete_executed_contracts
      ~SCML2022OneShotWorld.draw
      ~SCML2022OneShotWorld.executable_contracts
      ~SCML2022OneShotWorld.execute_action
      ~SCML2022OneShotWorld.from_checkpoint
      ~SCML2022OneShotWorld.from_config
      ~SCML2022OneShotWorld.generate
      ~SCML2022OneShotWorld.get_dropped_contracts
      ~SCML2022OneShotWorld.get_private_state
      ~SCML2022OneShotWorld.graph
      ~SCML2022OneShotWorld.ignore_contract
      ~SCML2022OneShotWorld.init
      ~SCML2022OneShotWorld.is_basic_stat
      ~SCML2022OneShotWorld.is_valid_agreement
      ~SCML2022OneShotWorld.is_valid_contact
      ~SCML2022OneShotWorld.is_valid_contract
      ~SCML2022OneShotWorld.join
      ~SCML2022OneShotWorld.logdebug
      ~SCML2022OneShotWorld.logdebug_agent
      ~SCML2022OneShotWorld.logerror
      ~SCML2022OneShotWorld.logerror_agent
      ~SCML2022OneShotWorld.loginfo
      ~SCML2022OneShotWorld.loginfo_agent
      ~SCML2022OneShotWorld.logwarning
      ~SCML2022OneShotWorld.logwarning_agent
      ~SCML2022OneShotWorld.n_saved_contracts
      ~SCML2022OneShotWorld.on_contract_cancelled
      ~SCML2022OneShotWorld.on_contract_concluded
      ~SCML2022OneShotWorld.on_contract_processed
      ~SCML2022OneShotWorld.on_contract_signed
      ~SCML2022OneShotWorld.on_event
      ~SCML2022OneShotWorld.on_exception
      ~SCML2022OneShotWorld.order_contracts_for_execution
      ~SCML2022OneShotWorld.post_step_stats
      ~SCML2022OneShotWorld.pre_step_stats
      ~SCML2022OneShotWorld.read_config
      ~SCML2022OneShotWorld.register
      ~SCML2022OneShotWorld.register_listener
      ~SCML2022OneShotWorld.register_stats_monitor
      ~SCML2022OneShotWorld.register_world_monitor
      ~SCML2022OneShotWorld.relative_welfare
      ~SCML2022OneShotWorld.request_negotiation_about
      ~SCML2022OneShotWorld.run
      ~SCML2022OneShotWorld.run_negotiation
      ~SCML2022OneShotWorld.run_negotiations
      ~SCML2022OneShotWorld.run_with_progress
      ~SCML2022OneShotWorld.save_config
      ~SCML2022OneShotWorld.save_gif
      ~SCML2022OneShotWorld.scores
      ~SCML2022OneShotWorld.set_bulletin_board
      ~SCML2022OneShotWorld.simulation_step
      ~SCML2022OneShotWorld.spawn
      ~SCML2022OneShotWorld.spawn_object
      ~SCML2022OneShotWorld.start_contract_execution
      ~SCML2022OneShotWorld.step
      ~SCML2022OneShotWorld.trading_prices_for
      ~SCML2022OneShotWorld.unregister_stats_monitor
      ~SCML2022OneShotWorld.unregister_world_monitor
      ~SCML2022OneShotWorld.update_stats
      ~SCML2022OneShotWorld.welfare

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
   .. autoattribute:: contracts_df
   .. autoattribute:: current_step
   .. autoattribute:: erred_contracts
   .. autoattribute:: executed_contracts
   .. autoattribute:: id
   .. autoattribute:: log_folder
   .. autoattribute:: n_agent_exceptions
   .. autoattribute:: n_contract_exceptions
   .. autoattribute:: n_mechanism_exceptions
   .. autoattribute:: n_negotiation_rounds_failed
   .. autoattribute:: n_negotiation_rounds_successful
   .. autoattribute:: n_negotiator_exceptions
   .. autoattribute:: n_simulation_exceptions
   .. autoattribute:: n_total_agent_exceptions
   .. autoattribute:: n_total_simulation_exceptions
   .. autoattribute:: name
   .. autoattribute:: non_system_agent_ids
   .. autoattribute:: non_system_agent_names
   .. autoattribute:: non_system_agents
   .. autoattribute:: nullified_contracts
   .. autoattribute:: relative_time
   .. autoattribute:: remaining_steps
   .. autoattribute:: remaining_time
   .. autoattribute:: resolved_breaches
   .. autoattribute:: saved_breaches
   .. autoattribute:: saved_contracts
   .. autoattribute:: saved_negotiations
   .. autoattribute:: short_type_name
   .. autoattribute:: signed_contracts
   .. autoattribute:: stats
   .. autoattribute:: stats_df
   .. autoattribute:: system_agent_ids
   .. autoattribute:: system_agent_names
   .. autoattribute:: system_agents
   .. autoattribute:: time
   .. autoattribute:: total_time
   .. autoattribute:: trading_prices
   .. autoattribute:: type_name
   .. autoattribute:: unresolved_breaches
   .. autoattribute:: uuid
   .. autoattribute:: winners

   .. rubric:: Methods Documentation

   .. automethod:: add_financial_report
   .. automethod:: announce
   .. automethod:: append_stats
   .. automethod:: breach_record
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_final_step
   .. automethod:: checkpoint_info
   .. automethod:: checkpoint_init
   .. automethod:: checkpoint_on_step_started
   .. automethod:: complete_contract_execution
   .. automethod:: contract_record
   .. automethod:: contract_size
   .. automethod:: create
   .. automethod:: current_balance
   .. automethod:: delete_executed_contracts
   .. automethod:: draw
   .. automethod:: executable_contracts
   .. automethod:: execute_action
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: generate
   .. automethod:: get_dropped_contracts
   .. automethod:: get_private_state
   .. automethod:: graph
   .. automethod:: ignore_contract
   .. automethod:: init
   .. automethod:: is_basic_stat
   .. automethod:: is_valid_agreement
   .. automethod:: is_valid_contact
   .. automethod:: is_valid_contract
   .. automethod:: join
   .. automethod:: logdebug
   .. automethod:: logdebug_agent
   .. automethod:: logerror
   .. automethod:: logerror_agent
   .. automethod:: loginfo
   .. automethod:: loginfo_agent
   .. automethod:: logwarning
   .. automethod:: logwarning_agent
   .. automethod:: n_saved_contracts
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_concluded
   .. automethod:: on_contract_processed
   .. automethod:: on_contract_signed
   .. automethod:: on_event
   .. automethod:: on_exception
   .. automethod:: order_contracts_for_execution
   .. automethod:: post_step_stats
   .. automethod:: pre_step_stats
   .. automethod:: read_config
   .. automethod:: register
   .. automethod:: register_listener
   .. automethod:: register_stats_monitor
   .. automethod:: register_world_monitor
   .. automethod:: relative_welfare
   .. automethod:: request_negotiation_about
   .. automethod:: run
   .. automethod:: run_negotiation
   .. automethod:: run_negotiations
   .. automethod:: run_with_progress
   .. automethod:: save_config
   .. automethod:: save_gif
   .. automethod:: scores
   .. automethod:: set_bulletin_board
   .. automethod:: simulation_step
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: start_contract_execution
   .. automethod:: step
   .. automethod:: trading_prices_for
   .. automethod:: unregister_stats_monitor
   .. automethod:: unregister_world_monitor
   .. automethod:: update_stats
   .. automethod:: welfare
