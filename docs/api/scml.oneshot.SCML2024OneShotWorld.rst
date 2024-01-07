SCML2024OneShotWorld
====================

.. currentmodule:: scml.oneshot

.. autoclass:: SCML2024OneShotWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2024OneShotWorld.agreement_fraction
      ~SCML2024OneShotWorld.agreement_rate
      ~SCML2024OneShotWorld.breach_fraction
      ~SCML2024OneShotWorld.breach_level
      ~SCML2024OneShotWorld.breach_rate
      ~SCML2024OneShotWorld.business_size
      ~SCML2024OneShotWorld.cancellation_fraction
      ~SCML2024OneShotWorld.cancellation_rate
      ~SCML2024OneShotWorld.cancelled_contracts
      ~SCML2024OneShotWorld.contract_dropping_fraction
      ~SCML2024OneShotWorld.contract_err_fraction
      ~SCML2024OneShotWorld.contract_execution_fraction
      ~SCML2024OneShotWorld.contract_nullification_fraction
      ~SCML2024OneShotWorld.contracts_df
      ~SCML2024OneShotWorld.current_step
      ~SCML2024OneShotWorld.erred_contracts
      ~SCML2024OneShotWorld.executed_contracts
      ~SCML2024OneShotWorld.id
      ~SCML2024OneShotWorld.log_folder
      ~SCML2024OneShotWorld.n_agent_exceptions
      ~SCML2024OneShotWorld.n_contract_exceptions
      ~SCML2024OneShotWorld.n_mechanism_exceptions
      ~SCML2024OneShotWorld.n_negotiation_rounds_failed
      ~SCML2024OneShotWorld.n_negotiation_rounds_successful
      ~SCML2024OneShotWorld.n_negotiator_exceptions
      ~SCML2024OneShotWorld.n_simulation_exceptions
      ~SCML2024OneShotWorld.n_total_agent_exceptions
      ~SCML2024OneShotWorld.n_total_simulation_exceptions
      ~SCML2024OneShotWorld.name
      ~SCML2024OneShotWorld.non_system_agent_ids
      ~SCML2024OneShotWorld.non_system_agent_names
      ~SCML2024OneShotWorld.non_system_agents
      ~SCML2024OneShotWorld.nullified_contracts
      ~SCML2024OneShotWorld.relative_time
      ~SCML2024OneShotWorld.remaining_steps
      ~SCML2024OneShotWorld.remaining_time
      ~SCML2024OneShotWorld.resolved_breaches
      ~SCML2024OneShotWorld.saved_breaches
      ~SCML2024OneShotWorld.saved_contracts
      ~SCML2024OneShotWorld.saved_negotiations
      ~SCML2024OneShotWorld.short_type_name
      ~SCML2024OneShotWorld.signed_contracts
      ~SCML2024OneShotWorld.stats
      ~SCML2024OneShotWorld.stats_df
      ~SCML2024OneShotWorld.system_agent_ids
      ~SCML2024OneShotWorld.system_agent_names
      ~SCML2024OneShotWorld.system_agents
      ~SCML2024OneShotWorld.time
      ~SCML2024OneShotWorld.total_time
      ~SCML2024OneShotWorld.trading_prices
      ~SCML2024OneShotWorld.type_name
      ~SCML2024OneShotWorld.unresolved_breaches
      ~SCML2024OneShotWorld.uuid
      ~SCML2024OneShotWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2024OneShotWorld.add_financial_report
      ~SCML2024OneShotWorld.announce
      ~SCML2024OneShotWorld.append_stats
      ~SCML2024OneShotWorld.breach_record
      ~SCML2024OneShotWorld.call
      ~SCML2024OneShotWorld.checkpoint
      ~SCML2024OneShotWorld.checkpoint_final_step
      ~SCML2024OneShotWorld.checkpoint_info
      ~SCML2024OneShotWorld.checkpoint_init
      ~SCML2024OneShotWorld.checkpoint_on_step_started
      ~SCML2024OneShotWorld.complete_contract_execution
      ~SCML2024OneShotWorld.contract_record
      ~SCML2024OneShotWorld.contract_size
      ~SCML2024OneShotWorld.create
      ~SCML2024OneShotWorld.current_balance
      ~SCML2024OneShotWorld.delete_executed_contracts
      ~SCML2024OneShotWorld.draw
      ~SCML2024OneShotWorld.executable_contracts
      ~SCML2024OneShotWorld.execute_action
      ~SCML2024OneShotWorld.from_checkpoint
      ~SCML2024OneShotWorld.from_config
      ~SCML2024OneShotWorld.generate
      ~SCML2024OneShotWorld.get_dropped_contracts
      ~SCML2024OneShotWorld.get_private_state
      ~SCML2024OneShotWorld.graph
      ~SCML2024OneShotWorld.ignore_contract
      ~SCML2024OneShotWorld.init
      ~SCML2024OneShotWorld.is_basic_stat
      ~SCML2024OneShotWorld.is_valid_agreement
      ~SCML2024OneShotWorld.is_valid_contact
      ~SCML2024OneShotWorld.is_valid_contract
      ~SCML2024OneShotWorld.join
      ~SCML2024OneShotWorld.logdebug
      ~SCML2024OneShotWorld.logdebug_agent
      ~SCML2024OneShotWorld.logerror
      ~SCML2024OneShotWorld.logerror_agent
      ~SCML2024OneShotWorld.loginfo
      ~SCML2024OneShotWorld.loginfo_agent
      ~SCML2024OneShotWorld.logwarning
      ~SCML2024OneShotWorld.logwarning_agent
      ~SCML2024OneShotWorld.n_saved_contracts
      ~SCML2024OneShotWorld.on_contract_cancelled
      ~SCML2024OneShotWorld.on_contract_concluded
      ~SCML2024OneShotWorld.on_contract_processed
      ~SCML2024OneShotWorld.on_contract_signed
      ~SCML2024OneShotWorld.on_event
      ~SCML2024OneShotWorld.on_exception
      ~SCML2024OneShotWorld.order_contracts_for_execution
      ~SCML2024OneShotWorld.post_step_stats
      ~SCML2024OneShotWorld.pre_step_stats
      ~SCML2024OneShotWorld.read_config
      ~SCML2024OneShotWorld.register
      ~SCML2024OneShotWorld.register_listener
      ~SCML2024OneShotWorld.register_stats_monitor
      ~SCML2024OneShotWorld.register_world_monitor
      ~SCML2024OneShotWorld.relative_welfare
      ~SCML2024OneShotWorld.request_negotiation_about
      ~SCML2024OneShotWorld.run
      ~SCML2024OneShotWorld.run_negotiation
      ~SCML2024OneShotWorld.run_negotiations
      ~SCML2024OneShotWorld.run_with_progress
      ~SCML2024OneShotWorld.save_config
      ~SCML2024OneShotWorld.save_gif
      ~SCML2024OneShotWorld.scores
      ~SCML2024OneShotWorld.set_bulletin_board
      ~SCML2024OneShotWorld.simulation_step
      ~SCML2024OneShotWorld.spawn
      ~SCML2024OneShotWorld.spawn_object
      ~SCML2024OneShotWorld.start_contract_execution
      ~SCML2024OneShotWorld.step
      ~SCML2024OneShotWorld.step_with
      ~SCML2024OneShotWorld.trading_prices_for
      ~SCML2024OneShotWorld.unregister_stats_monitor
      ~SCML2024OneShotWorld.unregister_world_monitor
      ~SCML2024OneShotWorld.update_stats
      ~SCML2024OneShotWorld.welfare

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
