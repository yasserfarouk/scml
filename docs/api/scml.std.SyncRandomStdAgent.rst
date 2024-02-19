SyncRandomStdAgent
==================

.. currentmodule:: scml.std

.. autoclass:: SyncRandomStdAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SyncRandomStdAgent.active_negotiators
      ~SyncRandomStdAgent.awi
      ~SyncRandomStdAgent.crisp_ufun
      ~SyncRandomStdAgent.has_cardinal_preferences
      ~SyncRandomStdAgent.has_preferences
      ~SyncRandomStdAgent.has_ufun
      ~SyncRandomStdAgent.id
      ~SyncRandomStdAgent.internal_state
      ~SyncRandomStdAgent.name
      ~SyncRandomStdAgent.negotiators
      ~SyncRandomStdAgent.preferences
      ~SyncRandomStdAgent.prob_ufun
      ~SyncRandomStdAgent.reserved_outcome
      ~SyncRandomStdAgent.reserved_value
      ~SyncRandomStdAgent.running_negotiations
      ~SyncRandomStdAgent.short_type_name
      ~SyncRandomStdAgent.states
      ~SyncRandomStdAgent.type_name
      ~SyncRandomStdAgent.type_postfix
      ~SyncRandomStdAgent.ufun
      ~SyncRandomStdAgent.unsigned_contracts
      ~SyncRandomStdAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~SyncRandomStdAgent.add_negotiator
      ~SyncRandomStdAgent.after_join
      ~SyncRandomStdAgent.before_join
      ~SyncRandomStdAgent.before_step
      ~SyncRandomStdAgent.best_price
      ~SyncRandomStdAgent.buy_price
      ~SyncRandomStdAgent.call
      ~SyncRandomStdAgent.checkpoint
      ~SyncRandomStdAgent.checkpoint_info
      ~SyncRandomStdAgent.connect_to_2021_adapter
      ~SyncRandomStdAgent.connect_to_oneshot_adapter
      ~SyncRandomStdAgent.counter_all
      ~SyncRandomStdAgent.create
      ~SyncRandomStdAgent.create_negotiator
      ~SyncRandomStdAgent.distribute_future_offers
      ~SyncRandomStdAgent.distribute_todays_needs
      ~SyncRandomStdAgent.estimate_future_needs
      ~SyncRandomStdAgent.first_offer
      ~SyncRandomStdAgent.first_proposals
      ~SyncRandomStdAgent.from_checkpoint
      ~SyncRandomStdAgent.get_ami
      ~SyncRandomStdAgent.get_negotiator
      ~SyncRandomStdAgent.get_nmi
      ~SyncRandomStdAgent.good2buy
      ~SyncRandomStdAgent.good2sell
      ~SyncRandomStdAgent.good_price
      ~SyncRandomStdAgent.init
      ~SyncRandomStdAgent.init_
      ~SyncRandomStdAgent.is_consumer
      ~SyncRandomStdAgent.is_supplier
      ~SyncRandomStdAgent.join
      ~SyncRandomStdAgent.kill_negotiator
      ~SyncRandomStdAgent.make_negotiator
      ~SyncRandomStdAgent.make_ufun
      ~SyncRandomStdAgent.on_contract_breached
      ~SyncRandomStdAgent.on_contract_executed
      ~SyncRandomStdAgent.on_leave
      ~SyncRandomStdAgent.on_mechanism_error
      ~SyncRandomStdAgent.on_negotiation_end
      ~SyncRandomStdAgent.on_negotiation_failure
      ~SyncRandomStdAgent.on_negotiation_start
      ~SyncRandomStdAgent.on_negotiation_success
      ~SyncRandomStdAgent.on_notification
      ~SyncRandomStdAgent.on_preferences_changed
      ~SyncRandomStdAgent.on_round_end
      ~SyncRandomStdAgent.on_round_start
      ~SyncRandomStdAgent.partner_agent_ids
      ~SyncRandomStdAgent.partner_agent_names
      ~SyncRandomStdAgent.partner_negotiator_ids
      ~SyncRandomStdAgent.partner_negotiator_names
      ~SyncRandomStdAgent.propose
      ~SyncRandomStdAgent.reset
      ~SyncRandomStdAgent.respond
      ~SyncRandomStdAgent.sell_price
      ~SyncRandomStdAgent.set_preferences
      ~SyncRandomStdAgent.sign_all_contracts
      ~SyncRandomStdAgent.spawn
      ~SyncRandomStdAgent.spawn_object
      ~SyncRandomStdAgent.step
      ~SyncRandomStdAgent.step_

   .. rubric:: Attributes Documentation

   .. autoattribute:: active_negotiators
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
   .. autoattribute:: has_ufun
   .. autoattribute:: id
   .. autoattribute:: internal_state
   .. autoattribute:: name
   .. autoattribute:: negotiators
   .. autoattribute:: preferences
   .. autoattribute:: prob_ufun
   .. autoattribute:: reserved_outcome
   .. autoattribute:: reserved_value
   .. autoattribute:: running_negotiations
   .. autoattribute:: short_type_name
   .. autoattribute:: states
   .. autoattribute:: type_name
   .. autoattribute:: type_postfix
   .. autoattribute:: ufun
   .. autoattribute:: unsigned_contracts
   .. autoattribute:: uuid

   .. rubric:: Methods Documentation

   .. automethod:: add_negotiator
   .. automethod:: after_join
   .. automethod:: before_join
   .. automethod:: before_step
   .. automethod:: best_price
   .. automethod:: buy_price
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: connect_to_2021_adapter
   .. automethod:: connect_to_oneshot_adapter
   .. automethod:: counter_all
   .. automethod:: create
   .. automethod:: create_negotiator
   .. automethod:: distribute_future_offers
   .. automethod:: distribute_todays_needs
   .. automethod:: estimate_future_needs
   .. automethod:: first_offer
   .. automethod:: first_proposals
   .. automethod:: from_checkpoint
   .. automethod:: get_ami
   .. automethod:: get_negotiator
   .. automethod:: get_nmi
   .. automethod:: good2buy
   .. automethod:: good2sell
   .. automethod:: good_price
   .. automethod:: init
   .. automethod:: init_
   .. automethod:: is_consumer
   .. automethod:: is_supplier
   .. automethod:: join
   .. automethod:: kill_negotiator
   .. automethod:: make_negotiator
   .. automethod:: make_ufun
   .. automethod:: on_contract_breached
   .. automethod:: on_contract_executed
   .. automethod:: on_leave
   .. automethod:: on_mechanism_error
   .. automethod:: on_negotiation_end
   .. automethod:: on_negotiation_failure
   .. automethod:: on_negotiation_start
   .. automethod:: on_negotiation_success
   .. automethod:: on_notification
   .. automethod:: on_preferences_changed
   .. automethod:: on_round_end
   .. automethod:: on_round_start
   .. automethod:: partner_agent_ids
   .. automethod:: partner_agent_names
   .. automethod:: partner_negotiator_ids
   .. automethod:: partner_negotiator_names
   .. automethod:: propose
   .. automethod:: reset
   .. automethod:: respond
   .. automethod:: sell_price
   .. automethod:: set_preferences
   .. automethod:: sign_all_contracts
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: step
   .. automethod:: step_
