SCML2021World
=============

.. currentmodule:: scml.scml2020

.. autoclass:: SCML2021World
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2021World.agreement_fraction
      ~SCML2021World.agreement_rate
      ~SCML2021World.bankruptcy_rate
      ~SCML2021World.breach_fraction
      ~SCML2021World.breach_level
      ~SCML2021World.breach_rate
      ~SCML2021World.business_size
      ~SCML2021World.cancellation_fraction
      ~SCML2021World.cancellation_rate
      ~SCML2021World.cancelled_contracts
      ~SCML2021World.contract_dropping_fraction
      ~SCML2021World.contract_err_fraction
      ~SCML2021World.contract_execution_fraction
      ~SCML2021World.contract_nullification_fraction
      ~SCML2021World.contracts_df
      ~SCML2021World.current_step
      ~SCML2021World.erred_contracts
      ~SCML2021World.executed_contracts
      ~SCML2021World.id
      ~SCML2021World.log_folder
      ~SCML2021World.n_agent_exceptions
      ~SCML2021World.n_contract_exceptions
      ~SCML2021World.n_mechanism_exceptions
      ~SCML2021World.n_negotiation_rounds_failed
      ~SCML2021World.n_negotiation_rounds_successful
      ~SCML2021World.n_negotiator_exceptions
      ~SCML2021World.n_simulation_exceptions
      ~SCML2021World.n_total_agent_exceptions
      ~SCML2021World.n_total_simulation_exceptions
      ~SCML2021World.name
      ~SCML2021World.non_system_agent_ids
      ~SCML2021World.non_system_agent_names
      ~SCML2021World.non_system_agents
      ~SCML2021World.nullified_contracts
      ~SCML2021World.num_bankrupt
      ~SCML2021World.productivity
      ~SCML2021World.relative_productivity
      ~SCML2021World.relative_time
      ~SCML2021World.remaining_steps
      ~SCML2021World.remaining_time
      ~SCML2021World.resolved_breaches
      ~SCML2021World.saved_breaches
      ~SCML2021World.saved_contracts
      ~SCML2021World.saved_negotiations
      ~SCML2021World.short_type_name
      ~SCML2021World.signed_contracts
      ~SCML2021World.stats
      ~SCML2021World.stats_df
      ~SCML2021World.system_agent_ids
      ~SCML2021World.system_agent_names
      ~SCML2021World.system_agents
      ~SCML2021World.time
      ~SCML2021World.total_time
      ~SCML2021World.trading_prices
      ~SCML2021World.type_name
      ~SCML2021World.unresolved_breaches
      ~SCML2021World.uuid
      ~SCML2021World.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2021World.add_financial_report
      ~SCML2021World.announce
      ~SCML2021World.append_stats
      ~SCML2021World.breach_record
      ~SCML2021World.call
      ~SCML2021World.can_negotiate
      ~SCML2021World.checkpoint
      ~SCML2021World.checkpoint_final_step
      ~SCML2021World.checkpoint_info
      ~SCML2021World.checkpoint_init
      ~SCML2021World.checkpoint_on_step_started
      ~SCML2021World.compensate
      ~SCML2021World.complete_contract_execution
      ~SCML2021World.contract_record
      ~SCML2021World.contract_size
      ~SCML2021World.create
      ~SCML2021World.current_balance
      ~SCML2021World.delete_executed_contracts
      ~SCML2021World.draw
      ~SCML2021World.executable_contracts
      ~SCML2021World.execute_action
      ~SCML2021World.from_checkpoint
      ~SCML2021World.from_config
      ~SCML2021World.generate
      ~SCML2021World.generate_guaranteed_profit
      ~SCML2021World.generate_profitable
      ~SCML2021World.get_dropped_contracts
      ~SCML2021World.get_private_state
      ~SCML2021World.graph
      ~SCML2021World.ignore_contract
      ~SCML2021World.init
      ~SCML2021World.is_basic_stat
      ~SCML2021World.is_valid_agreement
      ~SCML2021World.is_valid_contact
      ~SCML2021World.is_valid_contract
      ~SCML2021World.join
      ~SCML2021World.logdebug
      ~SCML2021World.logdebug_agent
      ~SCML2021World.logerror
      ~SCML2021World.logerror_agent
      ~SCML2021World.loginfo
      ~SCML2021World.loginfo_agent
      ~SCML2021World.logwarning
      ~SCML2021World.logwarning_agent
      ~SCML2021World.n_saved_contracts
      ~SCML2021World.negs_between
      ~SCML2021World.nullify_contract
      ~SCML2021World.on_contract_cancelled
      ~SCML2021World.on_contract_concluded
      ~SCML2021World.on_contract_processed
      ~SCML2021World.on_contract_signed
      ~SCML2021World.on_event
      ~SCML2021World.on_exception
      ~SCML2021World.order_contracts_for_execution
      ~SCML2021World.post_step_stats
      ~SCML2021World.pre_step_stats
      ~SCML2021World.read_config
      ~SCML2021World.record_bankrupt
      ~SCML2021World.register
      ~SCML2021World.register_listener
      ~SCML2021World.register_stats_monitor
      ~SCML2021World.register_world_monitor
      ~SCML2021World.relative_welfare
      ~SCML2021World.request_negotiation_about
      ~SCML2021World.run
      ~SCML2021World.run_negotiation
      ~SCML2021World.run_negotiations
      ~SCML2021World.run_with_progress
      ~SCML2021World.save_config
      ~SCML2021World.save_gif
      ~SCML2021World.scores
      ~SCML2021World.set_bulletin_board
      ~SCML2021World.simulation_step
      ~SCML2021World.spawn
      ~SCML2021World.spawn_object
      ~SCML2021World.start_contract_execution
      ~SCML2021World.step
      ~SCML2021World.trading_prices_for
      ~SCML2021World.unregister_stats_monitor
      ~SCML2021World.unregister_world_monitor
      ~SCML2021World.update_stats
      ~SCML2021World.welfare

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
