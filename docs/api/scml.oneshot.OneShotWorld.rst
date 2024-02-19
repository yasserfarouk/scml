OneShotWorld
============

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotWorld.agent_contracts
      ~OneShotWorld.agreement_fraction
      ~OneShotWorld.agreement_rate
      ~OneShotWorld.breach_fraction
      ~OneShotWorld.breach_level
      ~OneShotWorld.breach_rate
      ~OneShotWorld.business_size
      ~OneShotWorld.cancellation_fraction
      ~OneShotWorld.cancellation_rate
      ~OneShotWorld.cancelled_contracts
      ~OneShotWorld.contract_dropping_fraction
      ~OneShotWorld.contract_err_fraction
      ~OneShotWorld.contract_execution_fraction
      ~OneShotWorld.contract_nullification_fraction
      ~OneShotWorld.contracts_df
      ~OneShotWorld.current_step
      ~OneShotWorld.erred_contracts
      ~OneShotWorld.executed_contracts
      ~OneShotWorld.id
      ~OneShotWorld.log_folder
      ~OneShotWorld.n_agent_exceptions
      ~OneShotWorld.n_contract_exceptions
      ~OneShotWorld.n_mechanism_exceptions
      ~OneShotWorld.n_negotiation_rounds_failed
      ~OneShotWorld.n_negotiation_rounds_successful
      ~OneShotWorld.n_negotiator_exceptions
      ~OneShotWorld.n_simulation_exceptions
      ~OneShotWorld.n_total_agent_exceptions
      ~OneShotWorld.n_total_simulation_exceptions
      ~OneShotWorld.name
      ~OneShotWorld.non_system_agent_ids
      ~OneShotWorld.non_system_agent_names
      ~OneShotWorld.non_system_agents
      ~OneShotWorld.nullified_contracts
      ~OneShotWorld.relative_time
      ~OneShotWorld.remaining_steps
      ~OneShotWorld.remaining_time
      ~OneShotWorld.resolved_breaches
      ~OneShotWorld.saved_breaches
      ~OneShotWorld.saved_contracts
      ~OneShotWorld.saved_negotiations
      ~OneShotWorld.short_type_name
      ~OneShotWorld.signed_contracts
      ~OneShotWorld.stat_names
      ~OneShotWorld.stats
      ~OneShotWorld.stats_df
      ~OneShotWorld.system_agent_ids
      ~OneShotWorld.system_agent_names
      ~OneShotWorld.system_agents
      ~OneShotWorld.time
      ~OneShotWorld.total_time
      ~OneShotWorld.trading_prices
      ~OneShotWorld.type_name
      ~OneShotWorld.unresolved_breaches
      ~OneShotWorld.uuid
      ~OneShotWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotWorld.add_financial_report
      ~OneShotWorld.announce
      ~OneShotWorld.append_stats
      ~OneShotWorld.breach_record
      ~OneShotWorld.call
      ~OneShotWorld.checkpoint
      ~OneShotWorld.checkpoint_final_step
      ~OneShotWorld.checkpoint_info
      ~OneShotWorld.checkpoint_init
      ~OneShotWorld.checkpoint_on_step_started
      ~OneShotWorld.combine_stats
      ~OneShotWorld.complete_contract_execution
      ~OneShotWorld.contract_record
      ~OneShotWorld.contract_size
      ~OneShotWorld.create
      ~OneShotWorld.current_balance
      ~OneShotWorld.delete_executed_contracts
      ~OneShotWorld.draw
      ~OneShotWorld.executable_contracts
      ~OneShotWorld.execute_action
      ~OneShotWorld.from_checkpoint
      ~OneShotWorld.from_config
      ~OneShotWorld.generate
      ~OneShotWorld.get_dropped_contracts
      ~OneShotWorld.get_private_state
      ~OneShotWorld.graph
      ~OneShotWorld.ignore_contract
      ~OneShotWorld.init
      ~OneShotWorld.is_basic_stat
      ~OneShotWorld.is_valid_agreement
      ~OneShotWorld.is_valid_contact
      ~OneShotWorld.is_valid_contract
      ~OneShotWorld.join
      ~OneShotWorld.logdebug
      ~OneShotWorld.logdebug_agent
      ~OneShotWorld.logerror
      ~OneShotWorld.logerror_agent
      ~OneShotWorld.loginfo
      ~OneShotWorld.loginfo_agent
      ~OneShotWorld.logwarning
      ~OneShotWorld.logwarning_agent
      ~OneShotWorld.n_saved_contracts
      ~OneShotWorld.on_contract_cancelled
      ~OneShotWorld.on_contract_concluded
      ~OneShotWorld.on_contract_processed
      ~OneShotWorld.on_contract_signed
      ~OneShotWorld.on_event
      ~OneShotWorld.on_exception
      ~OneShotWorld.order_contracts_for_execution
      ~OneShotWorld.plot_combined_stats
      ~OneShotWorld.plot_stats
      ~OneShotWorld.post_step_stats
      ~OneShotWorld.pre_step_stats
      ~OneShotWorld.read_config
      ~OneShotWorld.register
      ~OneShotWorld.register_listener
      ~OneShotWorld.register_stats_monitor
      ~OneShotWorld.register_world_monitor
      ~OneShotWorld.relative_welfare
      ~OneShotWorld.replace_agents
      ~OneShotWorld.request_negotiation_about
      ~OneShotWorld.run
      ~OneShotWorld.run_negotiation
      ~OneShotWorld.run_negotiations
      ~OneShotWorld.run_with_progress
      ~OneShotWorld.save_config
      ~OneShotWorld.save_gif
      ~OneShotWorld.scores
      ~OneShotWorld.set_bulletin_board
      ~OneShotWorld.simulation_step
      ~OneShotWorld.spawn
      ~OneShotWorld.spawn_object
      ~OneShotWorld.start_contract_execution
      ~OneShotWorld.step
      ~OneShotWorld.step_with
      ~OneShotWorld.trading_prices_for
      ~OneShotWorld.unregister_stats_monitor
      ~OneShotWorld.unregister_world_monitor
      ~OneShotWorld.update_stats
      ~OneShotWorld.welfare

   .. rubric:: Attributes Documentation

   .. autoattribute:: agent_contracts
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
   .. autoattribute:: stat_names
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
   .. automethod:: combine_stats
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
   .. automethod:: plot_combined_stats
   .. automethod:: plot_stats
   .. automethod:: post_step_stats
   .. automethod:: pre_step_stats
   .. automethod:: read_config
   .. automethod:: register
   .. automethod:: register_listener
   .. automethod:: register_stats_monitor
   .. automethod:: register_world_monitor
   .. automethod:: relative_welfare
   .. automethod:: replace_agents
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
   .. automethod:: step_with
   .. automethod:: trading_prices_for
   .. automethod:: unregister_stats_monitor
   .. automethod:: unregister_world_monitor
   .. automethod:: update_stats
   .. automethod:: welfare
