SCML2021OneShotWorld
====================

.. currentmodule:: scml.oneshot

.. autoclass:: SCML2021OneShotWorld
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2021OneShotWorld.agreement_fraction
      ~SCML2021OneShotWorld.agreement_rate
      ~SCML2021OneShotWorld.breach_fraction
      ~SCML2021OneShotWorld.breach_level
      ~SCML2021OneShotWorld.breach_rate
      ~SCML2021OneShotWorld.business_size
      ~SCML2021OneShotWorld.cancellation_fraction
      ~SCML2021OneShotWorld.cancellation_rate
      ~SCML2021OneShotWorld.cancelled_contracts
      ~SCML2021OneShotWorld.contract_dropping_fraction
      ~SCML2021OneShotWorld.contract_err_fraction
      ~SCML2021OneShotWorld.contract_execution_fraction
      ~SCML2021OneShotWorld.contract_nullification_fraction
      ~SCML2021OneShotWorld.contracts_df
      ~SCML2021OneShotWorld.current_step
      ~SCML2021OneShotWorld.erred_contracts
      ~SCML2021OneShotWorld.executed_contracts
      ~SCML2021OneShotWorld.id
      ~SCML2021OneShotWorld.log_folder
      ~SCML2021OneShotWorld.n_agent_exceptions
      ~SCML2021OneShotWorld.n_contract_exceptions
      ~SCML2021OneShotWorld.n_mechanism_exceptions
      ~SCML2021OneShotWorld.n_negotiation_rounds_failed
      ~SCML2021OneShotWorld.n_negotiation_rounds_successful
      ~SCML2021OneShotWorld.n_negotiator_exceptions
      ~SCML2021OneShotWorld.n_simulation_exceptions
      ~SCML2021OneShotWorld.n_total_agent_exceptions
      ~SCML2021OneShotWorld.n_total_simulation_exceptions
      ~SCML2021OneShotWorld.name
      ~SCML2021OneShotWorld.non_system_agent_ids
      ~SCML2021OneShotWorld.non_system_agent_names
      ~SCML2021OneShotWorld.non_system_agents
      ~SCML2021OneShotWorld.nullified_contracts
      ~SCML2021OneShotWorld.relative_time
      ~SCML2021OneShotWorld.remaining_steps
      ~SCML2021OneShotWorld.remaining_time
      ~SCML2021OneShotWorld.resolved_breaches
      ~SCML2021OneShotWorld.saved_breaches
      ~SCML2021OneShotWorld.saved_contracts
      ~SCML2021OneShotWorld.saved_negotiations
      ~SCML2021OneShotWorld.short_type_name
      ~SCML2021OneShotWorld.signed_contracts
      ~SCML2021OneShotWorld.stats
      ~SCML2021OneShotWorld.stats_df
      ~SCML2021OneShotWorld.system_agent_ids
      ~SCML2021OneShotWorld.system_agent_names
      ~SCML2021OneShotWorld.system_agents
      ~SCML2021OneShotWorld.time
      ~SCML2021OneShotWorld.total_time
      ~SCML2021OneShotWorld.trading_prices
      ~SCML2021OneShotWorld.type_name
      ~SCML2021OneShotWorld.unresolved_breaches
      ~SCML2021OneShotWorld.uuid
      ~SCML2021OneShotWorld.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2021OneShotWorld.add_financial_report
      ~SCML2021OneShotWorld.announce
      ~SCML2021OneShotWorld.append_stats
      ~SCML2021OneShotWorld.breach_record
      ~SCML2021OneShotWorld.call
      ~SCML2021OneShotWorld.checkpoint
      ~SCML2021OneShotWorld.checkpoint_final_step
      ~SCML2021OneShotWorld.checkpoint_info
      ~SCML2021OneShotWorld.checkpoint_init
      ~SCML2021OneShotWorld.checkpoint_on_step_started
      ~SCML2021OneShotWorld.complete_contract_execution
      ~SCML2021OneShotWorld.contract_record
      ~SCML2021OneShotWorld.contract_size
      ~SCML2021OneShotWorld.create
      ~SCML2021OneShotWorld.current_balance
      ~SCML2021OneShotWorld.delete_executed_contracts
      ~SCML2021OneShotWorld.draw
      ~SCML2021OneShotWorld.executable_contracts
      ~SCML2021OneShotWorld.execute_action
      ~SCML2021OneShotWorld.from_checkpoint
      ~SCML2021OneShotWorld.from_config
      ~SCML2021OneShotWorld.generate
      ~SCML2021OneShotWorld.get_dropped_contracts
      ~SCML2021OneShotWorld.get_private_state
      ~SCML2021OneShotWorld.graph
      ~SCML2021OneShotWorld.ignore_contract
      ~SCML2021OneShotWorld.init
      ~SCML2021OneShotWorld.is_basic_stat
      ~SCML2021OneShotWorld.is_valid_agreement
      ~SCML2021OneShotWorld.is_valid_contact
      ~SCML2021OneShotWorld.is_valid_contract
      ~SCML2021OneShotWorld.join
      ~SCML2021OneShotWorld.logdebug
      ~SCML2021OneShotWorld.logdebug_agent
      ~SCML2021OneShotWorld.logerror
      ~SCML2021OneShotWorld.logerror_agent
      ~SCML2021OneShotWorld.loginfo
      ~SCML2021OneShotWorld.loginfo_agent
      ~SCML2021OneShotWorld.logwarning
      ~SCML2021OneShotWorld.logwarning_agent
      ~SCML2021OneShotWorld.n_saved_contracts
      ~SCML2021OneShotWorld.on_contract_cancelled
      ~SCML2021OneShotWorld.on_contract_concluded
      ~SCML2021OneShotWorld.on_contract_processed
      ~SCML2021OneShotWorld.on_contract_signed
      ~SCML2021OneShotWorld.on_event
      ~SCML2021OneShotWorld.on_exception
      ~SCML2021OneShotWorld.order_contracts_for_execution
      ~SCML2021OneShotWorld.post_step_stats
      ~SCML2021OneShotWorld.pre_step_stats
      ~SCML2021OneShotWorld.read_config
      ~SCML2021OneShotWorld.register
      ~SCML2021OneShotWorld.register_listener
      ~SCML2021OneShotWorld.register_stats_monitor
      ~SCML2021OneShotWorld.register_world_monitor
      ~SCML2021OneShotWorld.relative_welfare
      ~SCML2021OneShotWorld.request_negotiation_about
      ~SCML2021OneShotWorld.run
      ~SCML2021OneShotWorld.run_negotiation
      ~SCML2021OneShotWorld.run_negotiations
      ~SCML2021OneShotWorld.run_with_progress
      ~SCML2021OneShotWorld.save_config
      ~SCML2021OneShotWorld.save_gif
      ~SCML2021OneShotWorld.scores
      ~SCML2021OneShotWorld.set_bulletin_board
      ~SCML2021OneShotWorld.simulation_step
      ~SCML2021OneShotWorld.spawn
      ~SCML2021OneShotWorld.spawn_object
      ~SCML2021OneShotWorld.start_contract_execution
      ~SCML2021OneShotWorld.step
      ~SCML2021OneShotWorld.step_with
      ~SCML2021OneShotWorld.trading_prices_for
      ~SCML2021OneShotWorld.unregister_stats_monitor
      ~SCML2021OneShotWorld.unregister_world_monitor
      ~SCML2021OneShotWorld.update_stats
      ~SCML2021OneShotWorld.welfare

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
