SCML2023World
=============

.. currentmodule:: scml.scml2020

.. autoclass:: SCML2023World
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2023World.agreement_fraction
      ~SCML2023World.agreement_rate
      ~SCML2023World.bankruptcy_rate
      ~SCML2023World.breach_fraction
      ~SCML2023World.breach_level
      ~SCML2023World.breach_rate
      ~SCML2023World.business_size
      ~SCML2023World.cancellation_fraction
      ~SCML2023World.cancellation_rate
      ~SCML2023World.cancelled_contracts
      ~SCML2023World.contract_dropping_fraction
      ~SCML2023World.contract_err_fraction
      ~SCML2023World.contract_execution_fraction
      ~SCML2023World.contract_nullification_fraction
      ~SCML2023World.contracts_df
      ~SCML2023World.current_step
      ~SCML2023World.erred_contracts
      ~SCML2023World.executed_contracts
      ~SCML2023World.id
      ~SCML2023World.log_folder
      ~SCML2023World.n_agent_exceptions
      ~SCML2023World.n_contract_exceptions
      ~SCML2023World.n_mechanism_exceptions
      ~SCML2023World.n_negotiation_rounds_failed
      ~SCML2023World.n_negotiation_rounds_successful
      ~SCML2023World.n_negotiator_exceptions
      ~SCML2023World.n_simulation_exceptions
      ~SCML2023World.n_total_agent_exceptions
      ~SCML2023World.n_total_simulation_exceptions
      ~SCML2023World.name
      ~SCML2023World.non_system_agent_ids
      ~SCML2023World.non_system_agent_names
      ~SCML2023World.non_system_agents
      ~SCML2023World.nullified_contracts
      ~SCML2023World.num_bankrupt
      ~SCML2023World.productivity
      ~SCML2023World.relative_productivity
      ~SCML2023World.relative_time
      ~SCML2023World.remaining_steps
      ~SCML2023World.remaining_time
      ~SCML2023World.resolved_breaches
      ~SCML2023World.saved_breaches
      ~SCML2023World.saved_contracts
      ~SCML2023World.saved_negotiations
      ~SCML2023World.short_type_name
      ~SCML2023World.signed_contracts
      ~SCML2023World.stat_names
      ~SCML2023World.stats
      ~SCML2023World.stats_df
      ~SCML2023World.system_agent_ids
      ~SCML2023World.system_agent_names
      ~SCML2023World.system_agents
      ~SCML2023World.time
      ~SCML2023World.total_time
      ~SCML2023World.trading_prices
      ~SCML2023World.type_name
      ~SCML2023World.unresolved_breaches
      ~SCML2023World.uuid
      ~SCML2023World.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2023World.add_financial_report
      ~SCML2023World.announce
      ~SCML2023World.append_stats
      ~SCML2023World.breach_record
      ~SCML2023World.call
      ~SCML2023World.can_negotiate
      ~SCML2023World.checkpoint
      ~SCML2023World.checkpoint_final_step
      ~SCML2023World.checkpoint_info
      ~SCML2023World.checkpoint_init
      ~SCML2023World.checkpoint_on_step_started
      ~SCML2023World.combine_stats
      ~SCML2023World.compensate
      ~SCML2023World.complete_contract_execution
      ~SCML2023World.contract_record
      ~SCML2023World.contract_size
      ~SCML2023World.create
      ~SCML2023World.current_balance
      ~SCML2023World.delete_executed_contracts
      ~SCML2023World.draw
      ~SCML2023World.executable_contracts
      ~SCML2023World.execute_action
      ~SCML2023World.from_checkpoint
      ~SCML2023World.from_config
      ~SCML2023World.generate
      ~SCML2023World.generate_guaranteed_profit
      ~SCML2023World.generate_profitable
      ~SCML2023World.get_dropped_contracts
      ~SCML2023World.get_private_state
      ~SCML2023World.graph
      ~SCML2023World.ignore_contract
      ~SCML2023World.init
      ~SCML2023World.is_basic_stat
      ~SCML2023World.is_valid_agreement
      ~SCML2023World.is_valid_contact
      ~SCML2023World.is_valid_contract
      ~SCML2023World.join
      ~SCML2023World.logdebug
      ~SCML2023World.logdebug_agent
      ~SCML2023World.logerror
      ~SCML2023World.logerror_agent
      ~SCML2023World.loginfo
      ~SCML2023World.loginfo_agent
      ~SCML2023World.logwarning
      ~SCML2023World.logwarning_agent
      ~SCML2023World.n_saved_contracts
      ~SCML2023World.negs_between
      ~SCML2023World.nullify_contract
      ~SCML2023World.on_contract_cancelled
      ~SCML2023World.on_contract_concluded
      ~SCML2023World.on_contract_processed
      ~SCML2023World.on_contract_signed
      ~SCML2023World.on_event
      ~SCML2023World.on_exception
      ~SCML2023World.order_contracts_for_execution
      ~SCML2023World.plot_combined_stats
      ~SCML2023World.plot_stats
      ~SCML2023World.post_step_stats
      ~SCML2023World.pre_step_stats
      ~SCML2023World.read_config
      ~SCML2023World.record_bankrupt
      ~SCML2023World.register
      ~SCML2023World.register_listener
      ~SCML2023World.register_stats_monitor
      ~SCML2023World.register_world_monitor
      ~SCML2023World.relative_welfare
      ~SCML2023World.request_negotiation_about
      ~SCML2023World.run
      ~SCML2023World.run_negotiation
      ~SCML2023World.run_negotiations
      ~SCML2023World.run_with_progress
      ~SCML2023World.save_config
      ~SCML2023World.save_gif
      ~SCML2023World.scores
      ~SCML2023World.set_bulletin_board
      ~SCML2023World.simulation_step
      ~SCML2023World.spawn
      ~SCML2023World.spawn_object
      ~SCML2023World.start_contract_execution
      ~SCML2023World.step
      ~SCML2023World.trading_prices_for
      ~SCML2023World.unregister_stats_monitor
      ~SCML2023World.unregister_world_monitor
      ~SCML2023World.update_stats
      ~SCML2023World.welfare

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
