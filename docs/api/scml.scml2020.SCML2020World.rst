SCML2020World
=============

.. currentmodule:: scml.scml2020

.. autoclass:: SCML2020World
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SCML2020World.agreement_fraction
      ~SCML2020World.agreement_rate
      ~SCML2020World.bankruptcy_rate
      ~SCML2020World.breach_fraction
      ~SCML2020World.breach_level
      ~SCML2020World.breach_rate
      ~SCML2020World.business_size
      ~SCML2020World.cancellation_fraction
      ~SCML2020World.cancellation_rate
      ~SCML2020World.cancelled_contracts
      ~SCML2020World.contract_dropping_fraction
      ~SCML2020World.contract_err_fraction
      ~SCML2020World.contract_execution_fraction
      ~SCML2020World.contract_nullification_fraction
      ~SCML2020World.current_step
      ~SCML2020World.erred_contracts
      ~SCML2020World.executed_contracts
      ~SCML2020World.id
      ~SCML2020World.log_folder
      ~SCML2020World.n_negotiation_rounds_failed
      ~SCML2020World.n_negotiation_rounds_successful
      ~SCML2020World.name
      ~SCML2020World.nullified_contracts
      ~SCML2020World.num_bankrupt
      ~SCML2020World.productivity
      ~SCML2020World.relative_productivity
      ~SCML2020World.relative_time
      ~SCML2020World.remaining_steps
      ~SCML2020World.remaining_time
      ~SCML2020World.resolved_breaches
      ~SCML2020World.saved_breaches
      ~SCML2020World.saved_contracts
      ~SCML2020World.saved_negotiations
      ~SCML2020World.signed_contracts
      ~SCML2020World.stats
      ~SCML2020World.time
      ~SCML2020World.unresolved_breaches
      ~SCML2020World.uuid
      ~SCML2020World.winners

   .. rubric:: Methods Summary

   .. autosummary::

      ~SCML2020World.add_financial_report
      ~SCML2020World.announce
      ~SCML2020World.append_stats
      ~SCML2020World.breach_record
      ~SCML2020World.checkpoint
      ~SCML2020World.checkpoint_final_step
      ~SCML2020World.checkpoint_info
      ~SCML2020World.checkpoint_init
      ~SCML2020World.checkpoint_on_step_started
      ~SCML2020World.compensate
      ~SCML2020World.complete_contract_execution
      ~SCML2020World.contract_record
      ~SCML2020World.contract_size
      ~SCML2020World.create
      ~SCML2020World.delete_executed_contracts
      ~SCML2020World.executable_contracts
      ~SCML2020World.execute_action
      ~SCML2020World.from_checkpoint
      ~SCML2020World.from_config
      ~SCML2020World.generate
      ~SCML2020World.get_dropped_contracts
      ~SCML2020World.get_private_state
      ~SCML2020World.ignore_contract
      ~SCML2020World.init
      ~SCML2020World.join
      ~SCML2020World.logdebug
      ~SCML2020World.logdebug_agent
      ~SCML2020World.logerror
      ~SCML2020World.logerror_agent
      ~SCML2020World.loginfo
      ~SCML2020World.loginfo_agent
      ~SCML2020World.logwarning
      ~SCML2020World.logwarning_agent
      ~SCML2020World.n_saved_contracts
      ~SCML2020World.nullify_contract
      ~SCML2020World.on_contract_cancelled
      ~SCML2020World.on_contract_concluded
      ~SCML2020World.on_contract_processed
      ~SCML2020World.on_contract_signed
      ~SCML2020World.on_event
      ~SCML2020World.order_contracts_for_execution
      ~SCML2020World.post_step_stats
      ~SCML2020World.pre_step_stats
      ~SCML2020World.read_config
      ~SCML2020World.record_bankrupt
      ~SCML2020World.register
      ~SCML2020World.register_listener
      ~SCML2020World.register_stats_monitor
      ~SCML2020World.register_world_monitor
      ~SCML2020World.relative_welfare
      ~SCML2020World.request_negotiation_about
      ~SCML2020World.run
      ~SCML2020World.run_negotiation
      ~SCML2020World.run_negotiations
      ~SCML2020World.save_config
      ~SCML2020World.set_bulletin_board
      ~SCML2020World.simulation_step_after_execution
      ~SCML2020World.simulation_step_before_execution
      ~SCML2020World.start_contract_execution
      ~SCML2020World.step
      ~SCML2020World.unregister_stats_monitor
      ~SCML2020World.unregister_world_monitor
      ~SCML2020World.welfare

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
   .. autoattribute:: current_step
   .. autoattribute:: erred_contracts
   .. autoattribute:: executed_contracts
   .. autoattribute:: id
   .. autoattribute:: log_folder
   .. autoattribute:: n_negotiation_rounds_failed
   .. autoattribute:: n_negotiation_rounds_successful
   .. autoattribute:: name
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
   .. autoattribute:: signed_contracts
   .. autoattribute:: stats
   .. autoattribute:: time
   .. autoattribute:: unresolved_breaches
   .. autoattribute:: uuid
   .. autoattribute:: winners

   .. rubric:: Methods Documentation

   .. automethod:: add_financial_report
   .. automethod:: announce
   .. automethod:: append_stats
   .. automethod:: breach_record
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
   .. automethod:: delete_executed_contracts
   .. automethod:: executable_contracts
   .. automethod:: execute_action
   .. automethod:: from_checkpoint
   .. automethod:: from_config
   .. automethod:: generate
   .. automethod:: get_dropped_contracts
   .. automethod:: get_private_state
   .. automethod:: ignore_contract
   .. automethod:: init
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
   .. automethod:: nullify_contract
   .. automethod:: on_contract_cancelled
   .. automethod:: on_contract_concluded
   .. automethod:: on_contract_processed
   .. automethod:: on_contract_signed
   .. automethod:: on_event
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
   .. automethod:: save_config
   .. automethod:: set_bulletin_board
   .. automethod:: simulation_step_after_execution
   .. automethod:: simulation_step_before_execution
   .. automethod:: start_contract_execution
   .. automethod:: step
   .. automethod:: unregister_stats_monitor
   .. automethod:: unregister_world_monitor
   .. automethod:: welfare
