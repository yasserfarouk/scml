GreedyStdAgent
==============

.. currentmodule:: scml.std

.. autoclass:: GreedyStdAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~GreedyStdAgent.active_negotiators
      ~GreedyStdAgent.awi
      ~GreedyStdAgent.crisp_ufun
      ~GreedyStdAgent.has_cardinal_preferences
      ~GreedyStdAgent.has_preferences
      ~GreedyStdAgent.has_ufun
      ~GreedyStdAgent.id
      ~GreedyStdAgent.internal_state
      ~GreedyStdAgent.name
      ~GreedyStdAgent.negotiators
      ~GreedyStdAgent.preferences
      ~GreedyStdAgent.prob_ufun
      ~GreedyStdAgent.reserved_outcome
      ~GreedyStdAgent.reserved_value
      ~GreedyStdAgent.running_negotiations
      ~GreedyStdAgent.short_type_name
      ~GreedyStdAgent.states
      ~GreedyStdAgent.type_name
      ~GreedyStdAgent.type_postfix
      ~GreedyStdAgent.ufun
      ~GreedyStdAgent.unsigned_contracts
      ~GreedyStdAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~GreedyStdAgent.add_negotiator
      ~GreedyStdAgent.after_join
      ~GreedyStdAgent.before_join
      ~GreedyStdAgent.before_step
      ~GreedyStdAgent.best_offer
      ~GreedyStdAgent.call
      ~GreedyStdAgent.checkpoint
      ~GreedyStdAgent.checkpoint_info
      ~GreedyStdAgent.connect_to_2021_adapter
      ~GreedyStdAgent.connect_to_oneshot_adapter
      ~GreedyStdAgent.create
      ~GreedyStdAgent.create_negotiator
      ~GreedyStdAgent.from_checkpoint
      ~GreedyStdAgent.get_ami
      ~GreedyStdAgent.get_negotiator
      ~GreedyStdAgent.get_nmi
      ~GreedyStdAgent.init
      ~GreedyStdAgent.init_
      ~GreedyStdAgent.join
      ~GreedyStdAgent.kill_negotiator
      ~GreedyStdAgent.make_negotiator
      ~GreedyStdAgent.make_ufun
      ~GreedyStdAgent.on_contract_breached
      ~GreedyStdAgent.on_contract_executed
      ~GreedyStdAgent.on_leave
      ~GreedyStdAgent.on_mechanism_error
      ~GreedyStdAgent.on_negotiation_end
      ~GreedyStdAgent.on_negotiation_failure
      ~GreedyStdAgent.on_negotiation_start
      ~GreedyStdAgent.on_negotiation_success
      ~GreedyStdAgent.on_notification
      ~GreedyStdAgent.on_preferences_changed
      ~GreedyStdAgent.on_round_end
      ~GreedyStdAgent.on_round_start
      ~GreedyStdAgent.partner_agent_ids
      ~GreedyStdAgent.partner_agent_names
      ~GreedyStdAgent.partner_negotiator_ids
      ~GreedyStdAgent.partner_negotiator_names
      ~GreedyStdAgent.propose
      ~GreedyStdAgent.reset
      ~GreedyStdAgent.respond
      ~GreedyStdAgent.set_preferences
      ~GreedyStdAgent.sign_all_contracts
      ~GreedyStdAgent.spawn
      ~GreedyStdAgent.spawn_object
      ~GreedyStdAgent.step
      ~GreedyStdAgent.step_

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
   .. automethod:: best_offer
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: connect_to_2021_adapter
   .. automethod:: connect_to_oneshot_adapter
   .. automethod:: create
   .. automethod:: create_negotiator
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
