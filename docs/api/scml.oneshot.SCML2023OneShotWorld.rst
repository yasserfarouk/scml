SCML2023OneShotWorld
====================

.. currentmodule:: scml.oneshot

.. autoclass:: SCML2023OneShotWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2023OneShotWorld.agent_contracts
      ~SCML2023OneShotWorld.agreement_fraction
      ~SCML2023OneShotWorld.agreement_rate
      ~SCML2023OneShotWorld.breach_fraction
      ~SCML2023OneShotWorld.breach_level
      ~SCML2023OneShotWorld.breach_rate
      ~SCML2023OneShotWorld.business_size
      ~SCML2023OneShotWorld.cancellation_fraction
      ~SCML2023OneShotWorld.cancellation_rate
      ~SCML2023OneShotWorld.cancelled_contracts
      ~SCML2023OneShotWorld.contract_dropping_fraction
      ~SCML2023OneShotWorld.contract_err_fraction
      ~SCML2023OneShotWorld.contract_execution_fraction
      ~SCML2023OneShotWorld.contract_nullification_fraction
      ~SCML2023OneShotWorld.contracts_df
      ~SCML2023OneShotWorld.current_step
      ~SCML2023OneShotWorld.erred_contracts
      ~SCML2023OneShotWorld.executed_contracts
      ~SCML2023OneShotWorld.id
      ~SCML2023OneShotWorld.log_folder
      ~SCML2023OneShotWorld.n_agent_exceptions
      ~SCML2023OneShotWorld.n_contract_exceptions
      ~SCML2023OneShotWorld.n_mechanism_exceptions
      ~SCML2023OneShotWorld.n_negotiation_rounds_failed
      ~SCML2023OneShotWorld.n_negotiation_rounds_successful
      ~SCML2023OneShotWorld.n_negotiator_exceptions
      ~SCML2023OneShotWorld.n_simulation_exceptions
      ~SCML2023OneShotWorld.n_total_agent_exceptions
      ~SCML2023OneShotWorld.n_total_simulation_exceptions
      ~SCML2023OneShotWorld.name
      ~SCML2023OneShotWorld.non_system_agent_ids
      ~SCML2023OneShotWorld.non_system_agent_names
      ~SCML2023OneShotWorld.non_system_agents
      ~SCML2023OneShotWorld.nullified_contracts
      ~SCML2023OneShotWorld.relative_time
      ~SCML2023OneShotWorld.remaining_steps
      ~SCML2023OneShotWorld.remaining_time
      ~SCML2023OneShotWorld.resolved_breaches
      ~SCML2023OneShotWorld.saved_breaches
      ~SCML2023OneShotWorld.saved_contracts
      ~SCML2023OneShotWorld.saved_negotiations
      ~SCML2023OneShotWorld.short_type_name
      ~SCML2023OneShotWorld.signed_contracts
      ~SCML2023OneShotWorld.stat_names
      ~SCML2023OneShotWorld.stats
      ~SCML2023OneShotWorld.stats_df
      ~SCML2023OneShotWorld.system_agent_ids
      ~SCML2023OneShotWorld.system_agent_names
      ~SCML2023OneShotWorld.system_agents
      ~SCML2023OneShotWorld.time
      ~SCML2023OneShotWorld.total_time
      ~SCML2023OneShotWorld.trading_prices
      ~SCML2023OneShotWorld.type_name
      ~SCML2023OneShotWorld.unresolved_breaches
      ~SCML2023OneShotWorld.uuid
      ~SCML2023OneShotWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2023OneShotWorld.add_financial_report
      ~SCML2023OneShotWorld.announce
      ~SCML2023OneShotWorld.append_stats
      ~SCML2023OneShotWorld.breach_record
      ~SCML2023OneShotWorld.call
      ~SCML2023OneShotWorld.checkpoint
      ~SCML2023OneShotWorld.checkpoint_final_step
      ~SCML2023OneShotWorld.checkpoint_info
      ~SCML2023OneShotWorld.checkpoint_init
      ~SCML2023OneShotWorld.checkpoint_on_step_started
      ~SCML2023OneShotWorld.combine_stats
      ~SCML2023OneShotWorld.complete_contract_execution
      ~SCML2023OneShotWorld.contract_record
      ~SCML2023OneShotWorld.contract_size
      ~SCML2023OneShotWorld.create
      ~SCML2023OneShotWorld.current_balance
      ~SCML2023OneShotWorld.delete_executed_contracts
      ~SCML2023OneShotWorld.draw
      ~SCML2023OneShotWorld.executable_contracts
      ~SCML2023OneShotWorld.execute_action
      ~SCML2023OneShotWorld.from_checkpoint
      ~SCML2023OneShotWorld.from_config
      ~SCML2023OneShotWorld.generate
      ~SCML2023OneShotWorld.get_dropped_contracts
      ~SCML2023OneShotWorld.get_private_state
      ~SCML2023OneShotWorld.graph
      ~SCML2023OneShotWorld.ignore_contract
      ~SCML2023OneShotWorld.init
      ~SCML2023OneShotWorld.is_basic_stat
      ~SCML2023OneShotWorld.is_valid_agreement
      ~SCML2023OneShotWorld.is_valid_contact
      ~SCML2023OneShotWorld.is_valid_contract
      ~SCML2023OneShotWorld.join
      ~SCML2023OneShotWorld.logdebug
      ~SCML2023OneShotWorld.logdebug_agent
      ~SCML2023OneShotWorld.logerror
      ~SCML2023OneShotWorld.logerror_agent
      ~SCML2023OneShotWorld.loginfo
      ~SCML2023OneShotWorld.loginfo_agent
      ~SCML2023OneShotWorld.logwarning
      ~SCML2023OneShotWorld.logwarning_agent
      ~SCML2023OneShotWorld.n_saved_contracts
      ~SCML2023OneShotWorld.on_contract_cancelled
      ~SCML2023OneShotWorld.on_contract_concluded
      ~SCML2023OneShotWorld.on_contract_processed
      ~SCML2023OneShotWorld.on_contract_signed
      ~SCML2023OneShotWorld.on_event
      ~SCML2023OneShotWorld.on_exception
      ~SCML2023OneShotWorld.order_contracts_for_execution
      ~SCML2023OneShotWorld.plot_combined_stats
      ~SCML2023OneShotWorld.plot_stats
      ~SCML2023OneShotWorld.post_step_stats
      ~SCML2023OneShotWorld.pre_step_stats
      ~SCML2023OneShotWorld.read_config
      ~SCML2023OneShotWorld.register
      ~SCML2023OneShotWorld.register_listener
      ~SCML2023OneShotWorld.register_stats_monitor
      ~SCML2023OneShotWorld.register_world_monitor
      ~SCML2023OneShotWorld.relative_welfare
      ~SCML2023OneShotWorld.replace_agents
      ~SCML2023OneShotWorld.request_negotiation_about
      ~SCML2023OneShotWorld.run
      ~SCML2023OneShotWorld.run_negotiation
      ~SCML2023OneShotWorld.run_negotiations
      ~SCML2023OneShotWorld.run_with_progress
      ~SCML2023OneShotWorld.save_config
      ~SCML2023OneShotWorld.save_gif
      ~SCML2023OneShotWorld.scores
      ~SCML2023OneShotWorld.set_bulletin_board
      ~SCML2023OneShotWorld.simulation_step
      ~SCML2023OneShotWorld.spawn
      ~SCML2023OneShotWorld.spawn_object
      ~SCML2023OneShotWorld.start_contract_execution
      ~SCML2023OneShotWorld.step
      ~SCML2023OneShotWorld.step_with
      ~SCML2023OneShotWorld.trading_prices_for
      ~SCML2023OneShotWorld.unregister_stats_monitor
      ~SCML2023OneShotWorld.unregister_world_monitor
      ~SCML2023OneShotWorld.update_stats
      ~SCML2023OneShotWorld.welfare

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
