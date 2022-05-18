SCML2022World
=============

.. currentmodule:: scml.scml2020

.. autoclass:: SCML2022World
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2022World.agreement_fraction
      ~SCML2022World.agreement_rate
      ~SCML2022World.bankruptcy_rate
      ~SCML2022World.breach_fraction
      ~SCML2022World.breach_level
      ~SCML2022World.breach_rate
      ~SCML2022World.business_size
      ~SCML2022World.cancellation_fraction
      ~SCML2022World.cancellation_rate
      ~SCML2022World.cancelled_contracts
      ~SCML2022World.contract_dropping_fraction
      ~SCML2022World.contract_err_fraction
      ~SCML2022World.contract_execution_fraction
      ~SCML2022World.contract_nullification_fraction
      ~SCML2022World.contracts_df
      ~SCML2022World.current_step
      ~SCML2022World.erred_contracts
      ~SCML2022World.executed_contracts
      ~SCML2022World.id
      ~SCML2022World.log_folder
      ~SCML2022World.n_agent_exceptions
      ~SCML2022World.n_contract_exceptions
      ~SCML2022World.n_mechanism_exceptions
      ~SCML2022World.n_negotiation_rounds_failed
      ~SCML2022World.n_negotiation_rounds_successful
      ~SCML2022World.n_negotiator_exceptions
      ~SCML2022World.n_simulation_exceptions
      ~SCML2022World.n_total_agent_exceptions
      ~SCML2022World.n_total_simulation_exceptions
      ~SCML2022World.name
      ~SCML2022World.non_system_agent_ids
      ~SCML2022World.non_system_agent_names
      ~SCML2022World.non_system_agents
      ~SCML2022World.nullified_contracts
      ~SCML2022World.num_bankrupt
      ~SCML2022World.productivity
      ~SCML2022World.relative_productivity
      ~SCML2022World.relative_time
      ~SCML2022World.remaining_steps
      ~SCML2022World.remaining_time
      ~SCML2022World.resolved_breaches
      ~SCML2022World.saved_breaches
      ~SCML2022World.saved_contracts
      ~SCML2022World.saved_negotiations
      ~SCML2022World.short_type_name
      ~SCML2022World.signed_contracts
      ~SCML2022World.stats
      ~SCML2022World.stats_df
      ~SCML2022World.system_agent_ids
      ~SCML2022World.system_agent_names
      ~SCML2022World.system_agents
      ~SCML2022World.time
      ~SCML2022World.total_time
      ~SCML2022World.trading_prices
      ~SCML2022World.type_name
      ~SCML2022World.unresolved_breaches
      ~SCML2022World.uuid
      ~SCML2022World.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2022World.add_financial_report
      ~SCML2022World.announce
      ~SCML2022World.append_stats
      ~SCML2022World.breach_record
      ~SCML2022World.call
      ~SCML2022World.can_negotiate
      ~SCML2022World.checkpoint
      ~SCML2022World.checkpoint_final_step
      ~SCML2022World.checkpoint_info
      ~SCML2022World.checkpoint_init
      ~SCML2022World.checkpoint_on_step_started
      ~SCML2022World.compensate
      ~SCML2022World.complete_contract_execution
      ~SCML2022World.contract_record
      ~SCML2022World.contract_size
      ~SCML2022World.create
      ~SCML2022World.current_balance
      ~SCML2022World.delete_executed_contracts
      ~SCML2022World.draw
      ~SCML2022World.executable_contracts
      ~SCML2022World.execute_action
      ~SCML2022World.from_checkpoint
      ~SCML2022World.from_config
      ~SCML2022World.generate
      ~SCML2022World.generate_guaranteed_profit
      ~SCML2022World.generate_profitable
      ~SCML2022World.get_dropped_contracts
      ~SCML2022World.get_private_state
      ~SCML2022World.graph
      ~SCML2022World.ignore_contract
      ~SCML2022World.init
      ~SCML2022World.is_basic_stat
      ~SCML2022World.is_valid_agreement
      ~SCML2022World.is_valid_contact
      ~SCML2022World.is_valid_contract
      ~SCML2022World.join
      ~SCML2022World.logdebug
      ~SCML2022World.logdebug_agent
      ~SCML2022World.logerror
      ~SCML2022World.logerror_agent
      ~SCML2022World.loginfo
      ~SCML2022World.loginfo_agent
      ~SCML2022World.logwarning
      ~SCML2022World.logwarning_agent
      ~SCML2022World.n_saved_contracts
      ~SCML2022World.negs_between
      ~SCML2022World.nullify_contract
      ~SCML2022World.on_contract_cancelled
      ~SCML2022World.on_contract_concluded
      ~SCML2022World.on_contract_processed
      ~SCML2022World.on_contract_signed
      ~SCML2022World.on_event
      ~SCML2022World.on_exception
      ~SCML2022World.order_contracts_for_execution
      ~SCML2022World.post_step_stats
      ~SCML2022World.pre_step_stats
      ~SCML2022World.read_config
      ~SCML2022World.record_bankrupt
      ~SCML2022World.register
      ~SCML2022World.register_listener
      ~SCML2022World.register_stats_monitor
      ~SCML2022World.register_world_monitor
      ~SCML2022World.relative_welfare
      ~SCML2022World.request_negotiation_about
      ~SCML2022World.run
      ~SCML2022World.run_negotiation
      ~SCML2022World.run_negotiations
      ~SCML2022World.run_with_progress
      ~SCML2022World.save_config
      ~SCML2022World.save_gif
      ~SCML2022World.scores
      ~SCML2022World.set_bulletin_board
      ~SCML2022World.simulation_step
      ~SCML2022World.spawn
      ~SCML2022World.spawn_object
      ~SCML2022World.start_contract_execution
      ~SCML2022World.step
      ~SCML2022World.trading_prices_for
      ~SCML2022World.unregister_stats_monitor
      ~SCML2022World.unregister_world_monitor
      ~SCML2022World.update_stats
      ~SCML2022World.welfare

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
