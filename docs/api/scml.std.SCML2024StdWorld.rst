SCML2024StdWorld
================

.. currentmodule:: scml.std

.. autoclass:: SCML2024StdWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2024StdWorld.agreement_fraction
      ~SCML2024StdWorld.agreement_rate
      ~SCML2024StdWorld.breach_fraction
      ~SCML2024StdWorld.breach_level
      ~SCML2024StdWorld.breach_rate
      ~SCML2024StdWorld.business_size
      ~SCML2024StdWorld.cancellation_fraction
      ~SCML2024StdWorld.cancellation_rate
      ~SCML2024StdWorld.cancelled_contracts
      ~SCML2024StdWorld.contract_dropping_fraction
      ~SCML2024StdWorld.contract_err_fraction
      ~SCML2024StdWorld.contract_execution_fraction
      ~SCML2024StdWorld.contract_nullification_fraction
      ~SCML2024StdWorld.contracts_df
      ~SCML2024StdWorld.current_step
      ~SCML2024StdWorld.erred_contracts
      ~SCML2024StdWorld.executed_contracts
      ~SCML2024StdWorld.id
      ~SCML2024StdWorld.log_folder
      ~SCML2024StdWorld.n_agent_exceptions
      ~SCML2024StdWorld.n_contract_exceptions
      ~SCML2024StdWorld.n_mechanism_exceptions
      ~SCML2024StdWorld.n_negotiation_rounds_failed
      ~SCML2024StdWorld.n_negotiation_rounds_successful
      ~SCML2024StdWorld.n_negotiator_exceptions
      ~SCML2024StdWorld.n_simulation_exceptions
      ~SCML2024StdWorld.n_total_agent_exceptions
      ~SCML2024StdWorld.n_total_simulation_exceptions
      ~SCML2024StdWorld.name
      ~SCML2024StdWorld.non_system_agent_ids
      ~SCML2024StdWorld.non_system_agent_names
      ~SCML2024StdWorld.non_system_agents
      ~SCML2024StdWorld.nullified_contracts
      ~SCML2024StdWorld.relative_time
      ~SCML2024StdWorld.remaining_steps
      ~SCML2024StdWorld.remaining_time
      ~SCML2024StdWorld.resolved_breaches
      ~SCML2024StdWorld.saved_breaches
      ~SCML2024StdWorld.saved_contracts
      ~SCML2024StdWorld.saved_negotiations
      ~SCML2024StdWorld.short_type_name
      ~SCML2024StdWorld.signed_contracts
      ~SCML2024StdWorld.stats
      ~SCML2024StdWorld.stats_df
      ~SCML2024StdWorld.system_agent_ids
      ~SCML2024StdWorld.system_agent_names
      ~SCML2024StdWorld.system_agents
      ~SCML2024StdWorld.time
      ~SCML2024StdWorld.total_time
      ~SCML2024StdWorld.trading_prices
      ~SCML2024StdWorld.type_name
      ~SCML2024StdWorld.unresolved_breaches
      ~SCML2024StdWorld.uuid
      ~SCML2024StdWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2024StdWorld.add_financial_report
      ~SCML2024StdWorld.announce
      ~SCML2024StdWorld.append_stats
      ~SCML2024StdWorld.breach_record
      ~SCML2024StdWorld.call
      ~SCML2024StdWorld.checkpoint
      ~SCML2024StdWorld.checkpoint_final_step
      ~SCML2024StdWorld.checkpoint_info
      ~SCML2024StdWorld.checkpoint_init
      ~SCML2024StdWorld.checkpoint_on_step_started
      ~SCML2024StdWorld.complete_contract_execution
      ~SCML2024StdWorld.contract_record
      ~SCML2024StdWorld.contract_size
      ~SCML2024StdWorld.create
      ~SCML2024StdWorld.current_balance
      ~SCML2024StdWorld.delete_executed_contracts
      ~SCML2024StdWorld.draw
      ~SCML2024StdWorld.executable_contracts
      ~SCML2024StdWorld.execute_action
      ~SCML2024StdWorld.from_checkpoint
      ~SCML2024StdWorld.from_config
      ~SCML2024StdWorld.generate
      ~SCML2024StdWorld.get_dropped_contracts
      ~SCML2024StdWorld.get_private_state
      ~SCML2024StdWorld.graph
      ~SCML2024StdWorld.ignore_contract
      ~SCML2024StdWorld.init
      ~SCML2024StdWorld.is_basic_stat
      ~SCML2024StdWorld.is_valid_agreement
      ~SCML2024StdWorld.is_valid_contact
      ~SCML2024StdWorld.is_valid_contract
      ~SCML2024StdWorld.join
      ~SCML2024StdWorld.logdebug
      ~SCML2024StdWorld.logdebug_agent
      ~SCML2024StdWorld.logerror
      ~SCML2024StdWorld.logerror_agent
      ~SCML2024StdWorld.loginfo
      ~SCML2024StdWorld.loginfo_agent
      ~SCML2024StdWorld.logwarning
      ~SCML2024StdWorld.logwarning_agent
      ~SCML2024StdWorld.n_saved_contracts
      ~SCML2024StdWorld.on_contract_cancelled
      ~SCML2024StdWorld.on_contract_concluded
      ~SCML2024StdWorld.on_contract_processed
      ~SCML2024StdWorld.on_contract_signed
      ~SCML2024StdWorld.on_event
      ~SCML2024StdWorld.on_exception
      ~SCML2024StdWorld.order_contracts_for_execution
      ~SCML2024StdWorld.post_step_stats
      ~SCML2024StdWorld.pre_step_stats
      ~SCML2024StdWorld.read_config
      ~SCML2024StdWorld.register
      ~SCML2024StdWorld.register_listener
      ~SCML2024StdWorld.register_stats_monitor
      ~SCML2024StdWorld.register_world_monitor
      ~SCML2024StdWorld.relative_welfare
      ~SCML2024StdWorld.request_negotiation_about
      ~SCML2024StdWorld.run
      ~SCML2024StdWorld.run_negotiation
      ~SCML2024StdWorld.run_negotiations
      ~SCML2024StdWorld.run_with_progress
      ~SCML2024StdWorld.save_config
      ~SCML2024StdWorld.save_gif
      ~SCML2024StdWorld.scores
      ~SCML2024StdWorld.set_bulletin_board
      ~SCML2024StdWorld.simulation_step
      ~SCML2024StdWorld.spawn
      ~SCML2024StdWorld.spawn_object
      ~SCML2024StdWorld.start_contract_execution
      ~SCML2024StdWorld.step
      ~SCML2024StdWorld.step_with
      ~SCML2024StdWorld.trading_prices_for
      ~SCML2024StdWorld.unregister_stats_monitor
      ~SCML2024StdWorld.unregister_world_monitor
      ~SCML2024StdWorld.update_stats
      ~SCML2024StdWorld.welfare

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
