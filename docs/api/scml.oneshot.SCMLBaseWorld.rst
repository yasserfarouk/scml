SCMLBaseWorld
=============

.. currentmodule:: scml.oneshot

.. autoclass:: SCMLBaseWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCMLBaseWorld.agent_contracts
      ~SCMLBaseWorld.agreement_fraction
      ~SCMLBaseWorld.agreement_rate
      ~SCMLBaseWorld.breach_fraction
      ~SCMLBaseWorld.breach_level
      ~SCMLBaseWorld.breach_rate
      ~SCMLBaseWorld.business_size
      ~SCMLBaseWorld.cancellation_fraction
      ~SCMLBaseWorld.cancellation_rate
      ~SCMLBaseWorld.cancelled_contracts
      ~SCMLBaseWorld.contract_dropping_fraction
      ~SCMLBaseWorld.contract_err_fraction
      ~SCMLBaseWorld.contract_execution_fraction
      ~SCMLBaseWorld.contract_nullification_fraction
      ~SCMLBaseWorld.contracts_df
      ~SCMLBaseWorld.current_step
      ~SCMLBaseWorld.erred_contracts
      ~SCMLBaseWorld.executed_contracts
      ~SCMLBaseWorld.id
      ~SCMLBaseWorld.log_folder
      ~SCMLBaseWorld.n_agent_exceptions
      ~SCMLBaseWorld.n_contract_exceptions
      ~SCMLBaseWorld.n_mechanism_exceptions
      ~SCMLBaseWorld.n_negotiation_rounds_failed
      ~SCMLBaseWorld.n_negotiation_rounds_successful
      ~SCMLBaseWorld.n_negotiator_exceptions
      ~SCMLBaseWorld.n_simulation_exceptions
      ~SCMLBaseWorld.n_total_agent_exceptions
      ~SCMLBaseWorld.n_total_simulation_exceptions
      ~SCMLBaseWorld.name
      ~SCMLBaseWorld.non_system_agent_ids
      ~SCMLBaseWorld.non_system_agent_names
      ~SCMLBaseWorld.non_system_agents
      ~SCMLBaseWorld.nullified_contracts
      ~SCMLBaseWorld.relative_time
      ~SCMLBaseWorld.remaining_steps
      ~SCMLBaseWorld.remaining_time
      ~SCMLBaseWorld.resolved_breaches
      ~SCMLBaseWorld.saved_breaches
      ~SCMLBaseWorld.saved_contracts
      ~SCMLBaseWorld.saved_negotiations
      ~SCMLBaseWorld.short_type_name
      ~SCMLBaseWorld.signed_contracts
      ~SCMLBaseWorld.stat_names
      ~SCMLBaseWorld.stats
      ~SCMLBaseWorld.stats_df
      ~SCMLBaseWorld.system_agent_ids
      ~SCMLBaseWorld.system_agent_names
      ~SCMLBaseWorld.system_agents
      ~SCMLBaseWorld.time
      ~SCMLBaseWorld.total_time
      ~SCMLBaseWorld.trading_prices
      ~SCMLBaseWorld.type_name
      ~SCMLBaseWorld.unresolved_breaches
      ~SCMLBaseWorld.uuid
      ~SCMLBaseWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCMLBaseWorld.add_financial_report
      ~SCMLBaseWorld.announce
      ~SCMLBaseWorld.append_stats
      ~SCMLBaseWorld.breach_record
      ~SCMLBaseWorld.call
      ~SCMLBaseWorld.checkpoint
      ~SCMLBaseWorld.checkpoint_final_step
      ~SCMLBaseWorld.checkpoint_info
      ~SCMLBaseWorld.checkpoint_init
      ~SCMLBaseWorld.checkpoint_on_step_started
      ~SCMLBaseWorld.combine_stats
      ~SCMLBaseWorld.complete_contract_execution
      ~SCMLBaseWorld.contract_record
      ~SCMLBaseWorld.contract_size
      ~SCMLBaseWorld.create
      ~SCMLBaseWorld.current_balance
      ~SCMLBaseWorld.delete_executed_contracts
      ~SCMLBaseWorld.draw
      ~SCMLBaseWorld.executable_contracts
      ~SCMLBaseWorld.execute_action
      ~SCMLBaseWorld.from_checkpoint
      ~SCMLBaseWorld.from_config
      ~SCMLBaseWorld.generate
      ~SCMLBaseWorld.get_dropped_contracts
      ~SCMLBaseWorld.get_private_state
      ~SCMLBaseWorld.graph
      ~SCMLBaseWorld.ignore_contract
      ~SCMLBaseWorld.init
      ~SCMLBaseWorld.is_basic_stat
      ~SCMLBaseWorld.is_valid_agreement
      ~SCMLBaseWorld.is_valid_contact
      ~SCMLBaseWorld.is_valid_contract
      ~SCMLBaseWorld.join
      ~SCMLBaseWorld.logdebug
      ~SCMLBaseWorld.logdebug_agent
      ~SCMLBaseWorld.logerror
      ~SCMLBaseWorld.logerror_agent
      ~SCMLBaseWorld.loginfo
      ~SCMLBaseWorld.loginfo_agent
      ~SCMLBaseWorld.logwarning
      ~SCMLBaseWorld.logwarning_agent
      ~SCMLBaseWorld.n_saved_contracts
      ~SCMLBaseWorld.on_contract_cancelled
      ~SCMLBaseWorld.on_contract_concluded
      ~SCMLBaseWorld.on_contract_processed
      ~SCMLBaseWorld.on_contract_signed
      ~SCMLBaseWorld.on_event
      ~SCMLBaseWorld.on_exception
      ~SCMLBaseWorld.order_contracts_for_execution
      ~SCMLBaseWorld.plot_combined_stats
      ~SCMLBaseWorld.plot_stats
      ~SCMLBaseWorld.post_step_stats
      ~SCMLBaseWorld.pre_step_stats
      ~SCMLBaseWorld.read_config
      ~SCMLBaseWorld.register
      ~SCMLBaseWorld.register_listener
      ~SCMLBaseWorld.register_stats_monitor
      ~SCMLBaseWorld.register_world_monitor
      ~SCMLBaseWorld.relative_welfare
      ~SCMLBaseWorld.replace_agents
      ~SCMLBaseWorld.request_negotiation_about
      ~SCMLBaseWorld.run
      ~SCMLBaseWorld.run_negotiation
      ~SCMLBaseWorld.run_negotiations
      ~SCMLBaseWorld.run_with_progress
      ~SCMLBaseWorld.save_config
      ~SCMLBaseWorld.save_gif
      ~SCMLBaseWorld.scores
      ~SCMLBaseWorld.set_bulletin_board
      ~SCMLBaseWorld.simulation_step
      ~SCMLBaseWorld.spawn
      ~SCMLBaseWorld.spawn_object
      ~SCMLBaseWorld.start_contract_execution
      ~SCMLBaseWorld.step
      ~SCMLBaseWorld.step_with
      ~SCMLBaseWorld.trading_prices_for
      ~SCMLBaseWorld.unregister_stats_monitor
      ~SCMLBaseWorld.unregister_world_monitor
      ~SCMLBaseWorld.update_stats
      ~SCMLBaseWorld.welfare

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
