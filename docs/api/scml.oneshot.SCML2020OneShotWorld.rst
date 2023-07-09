SCML2020OneShotWorld
====================

.. currentmodule:: scml.oneshot

.. autoclass:: SCML2020OneShotWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2020OneShotWorld.agreement_fraction
      ~SCML2020OneShotWorld.agreement_rate
      ~SCML2020OneShotWorld.breach_fraction
      ~SCML2020OneShotWorld.breach_level
      ~SCML2020OneShotWorld.breach_rate
      ~SCML2020OneShotWorld.business_size
      ~SCML2020OneShotWorld.cancellation_fraction
      ~SCML2020OneShotWorld.cancellation_rate
      ~SCML2020OneShotWorld.cancelled_contracts
      ~SCML2020OneShotWorld.contract_dropping_fraction
      ~SCML2020OneShotWorld.contract_err_fraction
      ~SCML2020OneShotWorld.contract_execution_fraction
      ~SCML2020OneShotWorld.contract_nullification_fraction
      ~SCML2020OneShotWorld.contracts_df
      ~SCML2020OneShotWorld.current_step
      ~SCML2020OneShotWorld.erred_contracts
      ~SCML2020OneShotWorld.executed_contracts
      ~SCML2020OneShotWorld.id
      ~SCML2020OneShotWorld.log_folder
      ~SCML2020OneShotWorld.n_agent_exceptions
      ~SCML2020OneShotWorld.n_contract_exceptions
      ~SCML2020OneShotWorld.n_mechanism_exceptions
      ~SCML2020OneShotWorld.n_negotiation_rounds_failed
      ~SCML2020OneShotWorld.n_negotiation_rounds_successful
      ~SCML2020OneShotWorld.n_negotiator_exceptions
      ~SCML2020OneShotWorld.n_simulation_exceptions
      ~SCML2020OneShotWorld.n_total_agent_exceptions
      ~SCML2020OneShotWorld.n_total_simulation_exceptions
      ~SCML2020OneShotWorld.name
      ~SCML2020OneShotWorld.non_system_agent_ids
      ~SCML2020OneShotWorld.non_system_agent_names
      ~SCML2020OneShotWorld.non_system_agents
      ~SCML2020OneShotWorld.nullified_contracts
      ~SCML2020OneShotWorld.relative_time
      ~SCML2020OneShotWorld.remaining_steps
      ~SCML2020OneShotWorld.remaining_time
      ~SCML2020OneShotWorld.resolved_breaches
      ~SCML2020OneShotWorld.saved_breaches
      ~SCML2020OneShotWorld.saved_contracts
      ~SCML2020OneShotWorld.saved_negotiations
      ~SCML2020OneShotWorld.short_type_name
      ~SCML2020OneShotWorld.signed_contracts
      ~SCML2020OneShotWorld.stats
      ~SCML2020OneShotWorld.stats_df
      ~SCML2020OneShotWorld.system_agent_ids
      ~SCML2020OneShotWorld.system_agent_names
      ~SCML2020OneShotWorld.system_agents
      ~SCML2020OneShotWorld.time
      ~SCML2020OneShotWorld.total_time
      ~SCML2020OneShotWorld.trading_prices
      ~SCML2020OneShotWorld.type_name
      ~SCML2020OneShotWorld.unresolved_breaches
      ~SCML2020OneShotWorld.uuid
      ~SCML2020OneShotWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2020OneShotWorld.add_financial_report
      ~SCML2020OneShotWorld.announce
      ~SCML2020OneShotWorld.append_stats
      ~SCML2020OneShotWorld.breach_record
      ~SCML2020OneShotWorld.call
      ~SCML2020OneShotWorld.checkpoint
      ~SCML2020OneShotWorld.checkpoint_final_step
      ~SCML2020OneShotWorld.checkpoint_info
      ~SCML2020OneShotWorld.checkpoint_init
      ~SCML2020OneShotWorld.checkpoint_on_step_started
      ~SCML2020OneShotWorld.complete_contract_execution
      ~SCML2020OneShotWorld.contract_record
      ~SCML2020OneShotWorld.contract_size
      ~SCML2020OneShotWorld.create
      ~SCML2020OneShotWorld.current_balance
      ~SCML2020OneShotWorld.delete_executed_contracts
      ~SCML2020OneShotWorld.draw
      ~SCML2020OneShotWorld.executable_contracts
      ~SCML2020OneShotWorld.execute_action
      ~SCML2020OneShotWorld.from_checkpoint
      ~SCML2020OneShotWorld.from_config
      ~SCML2020OneShotWorld.generate
      ~SCML2020OneShotWorld.get_dropped_contracts
      ~SCML2020OneShotWorld.get_private_state
      ~SCML2020OneShotWorld.graph
      ~SCML2020OneShotWorld.ignore_contract
      ~SCML2020OneShotWorld.init
      ~SCML2020OneShotWorld.is_basic_stat
      ~SCML2020OneShotWorld.is_valid_agreement
      ~SCML2020OneShotWorld.is_valid_contact
      ~SCML2020OneShotWorld.is_valid_contract
      ~SCML2020OneShotWorld.join
      ~SCML2020OneShotWorld.logdebug
      ~SCML2020OneShotWorld.logdebug_agent
      ~SCML2020OneShotWorld.logerror
      ~SCML2020OneShotWorld.logerror_agent
      ~SCML2020OneShotWorld.loginfo
      ~SCML2020OneShotWorld.loginfo_agent
      ~SCML2020OneShotWorld.logwarning
      ~SCML2020OneShotWorld.logwarning_agent
      ~SCML2020OneShotWorld.n_saved_contracts
      ~SCML2020OneShotWorld.on_contract_cancelled
      ~SCML2020OneShotWorld.on_contract_concluded
      ~SCML2020OneShotWorld.on_contract_processed
      ~SCML2020OneShotWorld.on_contract_signed
      ~SCML2020OneShotWorld.on_event
      ~SCML2020OneShotWorld.on_exception
      ~SCML2020OneShotWorld.order_contracts_for_execution
      ~SCML2020OneShotWorld.post_step_stats
      ~SCML2020OneShotWorld.pre_step_stats
      ~SCML2020OneShotWorld.read_config
      ~SCML2020OneShotWorld.register
      ~SCML2020OneShotWorld.register_listener
      ~SCML2020OneShotWorld.register_stats_monitor
      ~SCML2020OneShotWorld.register_world_monitor
      ~SCML2020OneShotWorld.relative_welfare
      ~SCML2020OneShotWorld.request_negotiation_about
      ~SCML2020OneShotWorld.run
      ~SCML2020OneShotWorld.run_negotiation
      ~SCML2020OneShotWorld.run_negotiations
      ~SCML2020OneShotWorld.run_with_progress
      ~SCML2020OneShotWorld.save_config
      ~SCML2020OneShotWorld.save_gif
      ~SCML2020OneShotWorld.scores
      ~SCML2020OneShotWorld.set_bulletin_board
      ~SCML2020OneShotWorld.simulation_step
      ~SCML2020OneShotWorld.spawn
      ~SCML2020OneShotWorld.spawn_object
      ~SCML2020OneShotWorld.start_contract_execution
      ~SCML2020OneShotWorld.step
      ~SCML2020OneShotWorld.step_with
      ~SCML2020OneShotWorld.trading_prices_for
      ~SCML2020OneShotWorld.unregister_stats_monitor
      ~SCML2020OneShotWorld.unregister_world_monitor
      ~SCML2020OneShotWorld.update_stats
      ~SCML2020OneShotWorld.welfare

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
