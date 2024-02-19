SCML2024World
=============

.. currentmodule:: scml.scml2020

.. autoclass:: SCML2024World
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2024World.agreement_fraction
      ~SCML2024World.agreement_rate
      ~SCML2024World.bankruptcy_rate
      ~SCML2024World.breach_fraction
      ~SCML2024World.breach_level
      ~SCML2024World.breach_rate
      ~SCML2024World.business_size
      ~SCML2024World.cancellation_fraction
      ~SCML2024World.cancellation_rate
      ~SCML2024World.cancelled_contracts
      ~SCML2024World.contract_dropping_fraction
      ~SCML2024World.contract_err_fraction
      ~SCML2024World.contract_execution_fraction
      ~SCML2024World.contract_nullification_fraction
      ~SCML2024World.contracts_df
      ~SCML2024World.current_step
      ~SCML2024World.erred_contracts
      ~SCML2024World.executed_contracts
      ~SCML2024World.id
      ~SCML2024World.log_folder
      ~SCML2024World.n_agent_exceptions
      ~SCML2024World.n_contract_exceptions
      ~SCML2024World.n_mechanism_exceptions
      ~SCML2024World.n_negotiation_rounds_failed
      ~SCML2024World.n_negotiation_rounds_successful
      ~SCML2024World.n_negotiator_exceptions
      ~SCML2024World.n_simulation_exceptions
      ~SCML2024World.n_total_agent_exceptions
      ~SCML2024World.n_total_simulation_exceptions
      ~SCML2024World.name
      ~SCML2024World.non_system_agent_ids
      ~SCML2024World.non_system_agent_names
      ~SCML2024World.non_system_agents
      ~SCML2024World.nullified_contracts
      ~SCML2024World.num_bankrupt
      ~SCML2024World.productivity
      ~SCML2024World.relative_productivity
      ~SCML2024World.relative_time
      ~SCML2024World.remaining_steps
      ~SCML2024World.remaining_time
      ~SCML2024World.resolved_breaches
      ~SCML2024World.saved_breaches
      ~SCML2024World.saved_contracts
      ~SCML2024World.saved_negotiations
      ~SCML2024World.short_type_name
      ~SCML2024World.signed_contracts
      ~SCML2024World.stat_names
      ~SCML2024World.stats
      ~SCML2024World.stats_df
      ~SCML2024World.system_agent_ids
      ~SCML2024World.system_agent_names
      ~SCML2024World.system_agents
      ~SCML2024World.time
      ~SCML2024World.total_time
      ~SCML2024World.trading_prices
      ~SCML2024World.type_name
      ~SCML2024World.unresolved_breaches
      ~SCML2024World.uuid
      ~SCML2024World.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2024World.add_financial_report
      ~SCML2024World.announce
      ~SCML2024World.append_stats
      ~SCML2024World.breach_record
      ~SCML2024World.call
      ~SCML2024World.can_negotiate
      ~SCML2024World.checkpoint
      ~SCML2024World.checkpoint_final_step
      ~SCML2024World.checkpoint_info
      ~SCML2024World.checkpoint_init
      ~SCML2024World.checkpoint_on_step_started
      ~SCML2024World.combine_stats
      ~SCML2024World.compensate
      ~SCML2024World.complete_contract_execution
      ~SCML2024World.contract_record
      ~SCML2024World.contract_size
      ~SCML2024World.create
      ~SCML2024World.current_balance
      ~SCML2024World.delete_executed_contracts
      ~SCML2024World.draw
      ~SCML2024World.executable_contracts
      ~SCML2024World.execute_action
      ~SCML2024World.from_checkpoint
      ~SCML2024World.from_config
      ~SCML2024World.generate
      ~SCML2024World.generate_guaranteed_profit
      ~SCML2024World.generate_profitable
      ~SCML2024World.get_dropped_contracts
      ~SCML2024World.get_private_state
      ~SCML2024World.graph
      ~SCML2024World.ignore_contract
      ~SCML2024World.init
      ~SCML2024World.is_basic_stat
      ~SCML2024World.is_valid_agreement
      ~SCML2024World.is_valid_contact
      ~SCML2024World.is_valid_contract
      ~SCML2024World.join
      ~SCML2024World.logdebug
      ~SCML2024World.logdebug_agent
      ~SCML2024World.logerror
      ~SCML2024World.logerror_agent
      ~SCML2024World.loginfo
      ~SCML2024World.loginfo_agent
      ~SCML2024World.logwarning
      ~SCML2024World.logwarning_agent
      ~SCML2024World.n_saved_contracts
      ~SCML2024World.negs_between
      ~SCML2024World.nullify_contract
      ~SCML2024World.on_contract_cancelled
      ~SCML2024World.on_contract_concluded
      ~SCML2024World.on_contract_processed
      ~SCML2024World.on_contract_signed
      ~SCML2024World.on_event
      ~SCML2024World.on_exception
      ~SCML2024World.order_contracts_for_execution
      ~SCML2024World.plot_combined_stats
      ~SCML2024World.plot_stats
      ~SCML2024World.post_step_stats
      ~SCML2024World.pre_step_stats
      ~SCML2024World.read_config
      ~SCML2024World.record_bankrupt
      ~SCML2024World.register
      ~SCML2024World.register_listener
      ~SCML2024World.register_stats_monitor
      ~SCML2024World.register_world_monitor
      ~SCML2024World.relative_welfare
      ~SCML2024World.request_negotiation_about
      ~SCML2024World.run
      ~SCML2024World.run_negotiation
      ~SCML2024World.run_negotiations
      ~SCML2024World.run_with_progress
      ~SCML2024World.save_config
      ~SCML2024World.save_gif
      ~SCML2024World.scores
      ~SCML2024World.set_bulletin_board
      ~SCML2024World.simulation_step
      ~SCML2024World.spawn
      ~SCML2024World.spawn_object
      ~SCML2024World.start_contract_execution
      ~SCML2024World.step
      ~SCML2024World.trading_prices_for
      ~SCML2024World.unregister_stats_monitor
      ~SCML2024World.unregister_world_monitor
      ~SCML2024World.update_stats
      ~SCML2024World.welfare

   .. rubric:: Attributes Documentation

   .. autoattribute:: agreement_fraction
   .. autoattribute:: agreement_rate
   .. autoattribute:: bankruptcy_rate
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
   .. autoattribute:: num_bankrupt
   .. autoattribute:: productivity
   .. autoattribute:: relative_productivity
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
   .. automethod:: can_negotiate
   .. automethod:: checkpoint
   .. automethod:: checkpoint_final_step
   .. automethod:: checkpoint_info
   .. automethod:: checkpoint_init
   .. automethod:: checkpoint_on_step_started
   .. automethod:: combine_stats
   .. automethod:: compensate
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
   .. automethod:: generate_guaranteed_profit
   .. automethod:: generate_profitable
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
   .. automethod:: negs_between
   .. automethod:: nullify_contract
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
   .. automethod:: record_bankrupt
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
