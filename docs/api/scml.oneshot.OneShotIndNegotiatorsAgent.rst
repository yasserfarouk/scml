OneShotIndNegotiatorsAgent
==========================

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotIndNegotiatorsAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotIndNegotiatorsAgent.active_negotiators
      ~OneShotIndNegotiatorsAgent.awi
      ~OneShotIndNegotiatorsAgent.crisp_ufun
      ~OneShotIndNegotiatorsAgent.has_cardinal_preferences
      ~OneShotIndNegotiatorsAgent.has_preferences
      ~OneShotIndNegotiatorsAgent.has_ufun
      ~OneShotIndNegotiatorsAgent.id
      ~OneShotIndNegotiatorsAgent.internal_state
      ~OneShotIndNegotiatorsAgent.name
      ~OneShotIndNegotiatorsAgent.negotiators
      ~OneShotIndNegotiatorsAgent.preferences
      ~OneShotIndNegotiatorsAgent.prob_ufun
      ~OneShotIndNegotiatorsAgent.reserved_outcome
      ~OneShotIndNegotiatorsAgent.reserved_value
      ~OneShotIndNegotiatorsAgent.running_negotiations
      ~OneShotIndNegotiatorsAgent.short_type_name
      ~OneShotIndNegotiatorsAgent.states
      ~OneShotIndNegotiatorsAgent.type_name
      ~OneShotIndNegotiatorsAgent.type_postfix
      ~OneShotIndNegotiatorsAgent.ufun
      ~OneShotIndNegotiatorsAgent.unsigned_contracts
      ~OneShotIndNegotiatorsAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotIndNegotiatorsAgent.add_negotiator
      ~OneShotIndNegotiatorsAgent.after_join
      ~OneShotIndNegotiatorsAgent.before_join
      ~OneShotIndNegotiatorsAgent.before_step
      ~OneShotIndNegotiatorsAgent.call
      ~OneShotIndNegotiatorsAgent.checkpoint
      ~OneShotIndNegotiatorsAgent.checkpoint_info
      ~OneShotIndNegotiatorsAgent.connect_to_2021_adapter
      ~OneShotIndNegotiatorsAgent.connect_to_oneshot_adapter
      ~OneShotIndNegotiatorsAgent.create
      ~OneShotIndNegotiatorsAgent.create_negotiator
      ~OneShotIndNegotiatorsAgent.from_checkpoint
      ~OneShotIndNegotiatorsAgent.generate_negotiator
      ~OneShotIndNegotiatorsAgent.generate_ufuns
      ~OneShotIndNegotiatorsAgent.get_ami
      ~OneShotIndNegotiatorsAgent.get_negotiator
      ~OneShotIndNegotiatorsAgent.get_nmi
      ~OneShotIndNegotiatorsAgent.init
      ~OneShotIndNegotiatorsAgent.init_
      ~OneShotIndNegotiatorsAgent.join
      ~OneShotIndNegotiatorsAgent.kill_negotiator
      ~OneShotIndNegotiatorsAgent.make_negotiator
      ~OneShotIndNegotiatorsAgent.make_ufun
      ~OneShotIndNegotiatorsAgent.on_contract_breached
      ~OneShotIndNegotiatorsAgent.on_contract_executed
      ~OneShotIndNegotiatorsAgent.on_leave
      ~OneShotIndNegotiatorsAgent.on_mechanism_error
      ~OneShotIndNegotiatorsAgent.on_negotiation_end
      ~OneShotIndNegotiatorsAgent.on_negotiation_failure
      ~OneShotIndNegotiatorsAgent.on_negotiation_start
      ~OneShotIndNegotiatorsAgent.on_negotiation_success
      ~OneShotIndNegotiatorsAgent.on_notification
      ~OneShotIndNegotiatorsAgent.on_preferences_changed
      ~OneShotIndNegotiatorsAgent.on_round_end
      ~OneShotIndNegotiatorsAgent.on_round_start
      ~OneShotIndNegotiatorsAgent.partner_agent_ids
      ~OneShotIndNegotiatorsAgent.partner_agent_names
      ~OneShotIndNegotiatorsAgent.partner_negotiator_ids
      ~OneShotIndNegotiatorsAgent.partner_negotiator_names
      ~OneShotIndNegotiatorsAgent.propose
      ~OneShotIndNegotiatorsAgent.reset
      ~OneShotIndNegotiatorsAgent.respond
      ~OneShotIndNegotiatorsAgent.set_preferences
      ~OneShotIndNegotiatorsAgent.sign_all_contracts
      ~OneShotIndNegotiatorsAgent.spawn
      ~OneShotIndNegotiatorsAgent.spawn_object
      ~OneShotIndNegotiatorsAgent.step
      ~OneShotIndNegotiatorsAgent.step_

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
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: connect_to_2021_adapter
   .. automethod:: connect_to_oneshot_adapter
   .. automethod:: create
   .. automethod:: create_negotiator
   .. automethod:: from_checkpoint
   .. automethod:: generate_negotiator
   .. automethod:: generate_ufuns
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
