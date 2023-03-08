SyncRandomOneShotAgent
======================

.. currentmodule:: scml.oneshot

.. autoclass:: SyncRandomOneShotAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SyncRandomOneShotAgent.active_negotiators
      ~SyncRandomOneShotAgent.awi
      ~SyncRandomOneShotAgent.crisp_ufun
      ~SyncRandomOneShotAgent.has_cardinal_preferences
      ~SyncRandomOneShotAgent.has_preferences
      ~SyncRandomOneShotAgent.has_ufun
      ~SyncRandomOneShotAgent.id
      ~SyncRandomOneShotAgent.internal_state
      ~SyncRandomOneShotAgent.name
      ~SyncRandomOneShotAgent.negotiators
      ~SyncRandomOneShotAgent.preferences
      ~SyncRandomOneShotAgent.prob_ufun
      ~SyncRandomOneShotAgent.reserved_outcome
      ~SyncRandomOneShotAgent.reserved_value
      ~SyncRandomOneShotAgent.running_negotiations
      ~SyncRandomOneShotAgent.short_type_name
      ~SyncRandomOneShotAgent.states
      ~SyncRandomOneShotAgent.type_name
      ~SyncRandomOneShotAgent.type_postfix
      ~SyncRandomOneShotAgent.ufun
      ~SyncRandomOneShotAgent.unsigned_contracts
      ~SyncRandomOneShotAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~SyncRandomOneShotAgent.add_negotiator
      ~SyncRandomOneShotAgent.after_join
      ~SyncRandomOneShotAgent.before_join
      ~SyncRandomOneShotAgent.call
      ~SyncRandomOneShotAgent.checkpoint
      ~SyncRandomOneShotAgent.checkpoint_info
      ~SyncRandomOneShotAgent.connect_to_2021_adapter
      ~SyncRandomOneShotAgent.connect_to_oneshot_adapter
      ~SyncRandomOneShotAgent.counter_all
      ~SyncRandomOneShotAgent.create
      ~SyncRandomOneShotAgent.create_negotiator
      ~SyncRandomOneShotAgent.first_offer
      ~SyncRandomOneShotAgent.first_proposals
      ~SyncRandomOneShotAgent.from_checkpoint
      ~SyncRandomOneShotAgent.get_ami
      ~SyncRandomOneShotAgent.get_negotiator
      ~SyncRandomOneShotAgent.get_nmi
      ~SyncRandomOneShotAgent.init
      ~SyncRandomOneShotAgent.init_
      ~SyncRandomOneShotAgent.join
      ~SyncRandomOneShotAgent.kill_negotiator
      ~SyncRandomOneShotAgent.make_negotiator
      ~SyncRandomOneShotAgent.make_ufun
      ~SyncRandomOneShotAgent.on_contract_breached
      ~SyncRandomOneShotAgent.on_contract_executed
      ~SyncRandomOneShotAgent.on_leave
      ~SyncRandomOneShotAgent.on_mechanism_error
      ~SyncRandomOneShotAgent.on_negotiation_end
      ~SyncRandomOneShotAgent.on_negotiation_failure
      ~SyncRandomOneShotAgent.on_negotiation_start
      ~SyncRandomOneShotAgent.on_negotiation_success
      ~SyncRandomOneShotAgent.on_notification
      ~SyncRandomOneShotAgent.on_preferences_changed
      ~SyncRandomOneShotAgent.on_round_end
      ~SyncRandomOneShotAgent.on_round_start
      ~SyncRandomOneShotAgent.partner_agent_ids
      ~SyncRandomOneShotAgent.partner_agent_names
      ~SyncRandomOneShotAgent.partner_negotiator_ids
      ~SyncRandomOneShotAgent.partner_negotiator_names
      ~SyncRandomOneShotAgent.propose
      ~SyncRandomOneShotAgent.reset
      ~SyncRandomOneShotAgent.respond
      ~SyncRandomOneShotAgent.set_preferences
      ~SyncRandomOneShotAgent.sign_all_contracts
      ~SyncRandomOneShotAgent.spawn
      ~SyncRandomOneShotAgent.spawn_object
      ~SyncRandomOneShotAgent.step
      ~SyncRandomOneShotAgent.step_

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
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: connect_to_2021_adapter
   .. automethod:: connect_to_oneshot_adapter
   .. automethod:: counter_all
   .. automethod:: create
   .. automethod:: create_negotiator
   .. automethod:: first_offer
   .. automethod:: first_proposals
   .. automethod:: from_checkpoint
   .. automethod:: get_ami
   .. automethod:: get_negotiator
   .. automethod:: get_nmi
   .. automethod:: init
   .. automethod:: init_
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
   .. automethod:: set_preferences
   .. automethod:: sign_all_contracts
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: step
   .. automethod:: step_
