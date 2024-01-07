StdWorld
========

.. currentmodule:: scml.std

.. autoclass:: StdWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~StdWorld.agreement_fraction
      ~StdWorld.agreement_rate
      ~StdWorld.breach_fraction
      ~StdWorld.breach_level
      ~StdWorld.breach_rate
      ~StdWorld.business_size
      ~StdWorld.cancellation_fraction
      ~StdWorld.cancellation_rate
      ~StdWorld.cancelled_contracts
      ~StdWorld.contract_dropping_fraction
      ~StdWorld.contract_err_fraction
      ~StdWorld.contract_execution_fraction
      ~StdWorld.contract_nullification_fraction
      ~StdWorld.contracts_df
      ~StdWorld.current_step
      ~StdWorld.erred_contracts
      ~StdWorld.executed_contracts
      ~StdWorld.id
      ~StdWorld.log_folder
      ~StdWorld.n_agent_exceptions
      ~StdWorld.n_contract_exceptions
      ~StdWorld.n_mechanism_exceptions
      ~StdWorld.n_negotiation_rounds_failed
      ~StdWorld.n_negotiation_rounds_successful
      ~StdWorld.n_negotiator_exceptions
      ~StdWorld.n_simulation_exceptions
      ~StdWorld.n_total_agent_exceptions
      ~StdWorld.n_total_simulation_exceptions
      ~StdWorld.name
      ~StdWorld.non_system_agent_ids
      ~StdWorld.non_system_agent_names
      ~StdWorld.non_system_agents
      ~StdWorld.nullified_contracts
      ~StdWorld.relative_time
      ~StdWorld.remaining_steps
      ~StdWorld.remaining_time
      ~StdWorld.resolved_breaches
      ~StdWorld.saved_breaches
      ~StdWorld.saved_contracts
      ~StdWorld.saved_negotiations
      ~StdWorld.short_type_name
      ~StdWorld.signed_contracts
      ~StdWorld.stats
      ~StdWorld.stats_df
      ~StdWorld.system_agent_ids
      ~StdWorld.system_agent_names
      ~StdWorld.system_agents
      ~StdWorld.time
      ~StdWorld.total_time
      ~StdWorld.trading_prices
      ~StdWorld.type_name
      ~StdWorld.unresolved_breaches
      ~StdWorld.uuid
      ~StdWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~StdWorld.add_financial_report
      ~StdWorld.announce
      ~StdWorld.append_stats
      ~StdWorld.breach_record
      ~StdWorld.call
      ~StdWorld.checkpoint
      ~StdWorld.checkpoint_final_step
      ~StdWorld.checkpoint_info
      ~StdWorld.checkpoint_init
      ~StdWorld.checkpoint_on_step_started
      ~StdWorld.complete_contract_execution
      ~StdWorld.contract_record
      ~StdWorld.contract_size
      ~StdWorld.create
      ~StdWorld.current_balance
      ~StdWorld.delete_executed_contracts
      ~StdWorld.draw
      ~StdWorld.executable_contracts
      ~StdWorld.execute_action
      ~StdWorld.from_checkpoint
      ~StdWorld.from_config
      ~StdWorld.generate
      ~StdWorld.get_dropped_contracts
      ~StdWorld.get_private_state
      ~StdWorld.graph
      ~StdWorld.ignore_contract
      ~StdWorld.init
      ~StdWorld.is_basic_stat
      ~StdWorld.is_valid_agreement
      ~StdWorld.is_valid_contact
      ~StdWorld.is_valid_contract
      ~StdWorld.join
      ~StdWorld.logdebug
      ~StdWorld.logdebug_agent
      ~StdWorld.logerror
      ~StdWorld.logerror_agent
      ~StdWorld.loginfo
      ~StdWorld.loginfo_agent
      ~StdWorld.logwarning
      ~StdWorld.logwarning_agent
      ~StdWorld.n_saved_contracts
      ~StdWorld.on_contract_cancelled
      ~StdWorld.on_contract_concluded
      ~StdWorld.on_contract_processed
      ~StdWorld.on_contract_signed
      ~StdWorld.on_event
      ~StdWorld.on_exception
      ~StdWorld.order_contracts_for_execution
      ~StdWorld.post_step_stats
      ~StdWorld.pre_step_stats
      ~StdWorld.read_config
      ~StdWorld.register
      ~StdWorld.register_listener
      ~StdWorld.register_stats_monitor
      ~StdWorld.register_world_monitor
      ~StdWorld.relative_welfare
      ~StdWorld.request_negotiation_about
      ~StdWorld.run
      ~StdWorld.run_negotiation
      ~StdWorld.run_negotiations
      ~StdWorld.run_with_progress
      ~StdWorld.save_config
      ~StdWorld.save_gif
      ~StdWorld.scores
      ~StdWorld.set_bulletin_board
      ~StdWorld.simulation_step
      ~StdWorld.spawn
      ~StdWorld.spawn_object
      ~StdWorld.start_contract_execution
      ~StdWorld.step
      ~StdWorld.step_with
      ~StdWorld.trading_prices_for
      ~StdWorld.unregister_stats_monitor
      ~StdWorld.unregister_world_monitor
      ~StdWorld.update_stats
      ~StdWorld.welfare

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
   .. automethod:: step_with
   .. automethod:: trading_prices_for
   .. automethod:: unregister_stats_monitor
   .. automethod:: unregister_world_monitor
   .. automethod:: update_stats
   .. automethod:: welfare
