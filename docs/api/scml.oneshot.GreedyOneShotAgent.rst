GreedyOneShotAgent
==================

.. currentmodule:: scml.oneshot

.. autoclass:: GreedyOneShotAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~GreedyOneShotAgent.active_negotiators
      ~GreedyOneShotAgent.awi
      ~GreedyOneShotAgent.crisp_ufun
      ~GreedyOneShotAgent.has_cardinal_preferences
      ~GreedyOneShotAgent.has_preferences
      ~GreedyOneShotAgent.has_ufun
      ~GreedyOneShotAgent.id
      ~GreedyOneShotAgent.internal_state
      ~GreedyOneShotAgent.name
      ~GreedyOneShotAgent.negotiators
      ~GreedyOneShotAgent.preferences
      ~GreedyOneShotAgent.prob_ufun
      ~GreedyOneShotAgent.reserved_outcome
      ~GreedyOneShotAgent.reserved_value
      ~GreedyOneShotAgent.running_negotiations
      ~GreedyOneShotAgent.short_type_name
      ~GreedyOneShotAgent.states
      ~GreedyOneShotAgent.type_name
      ~GreedyOneShotAgent.type_postfix
      ~GreedyOneShotAgent.ufun
      ~GreedyOneShotAgent.unsigned_contracts
      ~GreedyOneShotAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~GreedyOneShotAgent.add_negotiator
      ~GreedyOneShotAgent.after_join
      ~GreedyOneShotAgent.before_join
      ~GreedyOneShotAgent.before_step
      ~GreedyOneShotAgent.best_offer
      ~GreedyOneShotAgent.call
      ~GreedyOneShotAgent.checkpoint
      ~GreedyOneShotAgent.checkpoint_info
      ~GreedyOneShotAgent.connect_to_2021_adapter
      ~GreedyOneShotAgent.connect_to_oneshot_adapter
      ~GreedyOneShotAgent.create
      ~GreedyOneShotAgent.create_negotiator
      ~GreedyOneShotAgent.from_checkpoint
      ~GreedyOneShotAgent.get_ami
      ~GreedyOneShotAgent.get_negotiator
      ~GreedyOneShotAgent.get_nmi
      ~GreedyOneShotAgent.init
      ~GreedyOneShotAgent.init_
      ~GreedyOneShotAgent.join
      ~GreedyOneShotAgent.kill_negotiator
      ~GreedyOneShotAgent.make_negotiator
      ~GreedyOneShotAgent.make_ufun
      ~GreedyOneShotAgent.on_contract_breached
      ~GreedyOneShotAgent.on_contract_executed
      ~GreedyOneShotAgent.on_leave
      ~GreedyOneShotAgent.on_mechanism_error
      ~GreedyOneShotAgent.on_negotiation_end
      ~GreedyOneShotAgent.on_negotiation_failure
      ~GreedyOneShotAgent.on_negotiation_start
      ~GreedyOneShotAgent.on_negotiation_success
      ~GreedyOneShotAgent.on_notification
      ~GreedyOneShotAgent.on_preferences_changed
      ~GreedyOneShotAgent.on_round_end
      ~GreedyOneShotAgent.on_round_start
      ~GreedyOneShotAgent.partner_agent_ids
      ~GreedyOneShotAgent.partner_agent_names
      ~GreedyOneShotAgent.partner_negotiator_ids
      ~GreedyOneShotAgent.partner_negotiator_names
      ~GreedyOneShotAgent.propose
      ~GreedyOneShotAgent.reset
      ~GreedyOneShotAgent.respond
      ~GreedyOneShotAgent.set_preferences
      ~GreedyOneShotAgent.sign_all_contracts
      ~GreedyOneShotAgent.spawn
      ~GreedyOneShotAgent.spawn_object
      ~GreedyOneShotAgent.step
      ~GreedyOneShotAgent.step_

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
