GreedySyncAgent
===============

.. currentmodule:: scml.oneshot

.. autoclass:: GreedySyncAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~GreedySyncAgent.active_negotiators
      ~GreedySyncAgent.awi
      ~GreedySyncAgent.crisp_ufun
      ~GreedySyncAgent.has_cardinal_preferences
      ~GreedySyncAgent.has_preferences
      ~GreedySyncAgent.has_ufun
      ~GreedySyncAgent.id
      ~GreedySyncAgent.internal_state
      ~GreedySyncAgent.name
      ~GreedySyncAgent.negotiators
      ~GreedySyncAgent.preferences
      ~GreedySyncAgent.prob_ufun
      ~GreedySyncAgent.reserved_outcome
      ~GreedySyncAgent.reserved_value
      ~GreedySyncAgent.running_negotiations
      ~GreedySyncAgent.short_type_name
      ~GreedySyncAgent.states
      ~GreedySyncAgent.type_name
      ~GreedySyncAgent.type_postfix
      ~GreedySyncAgent.ufun
      ~GreedySyncAgent.unsigned_contracts
      ~GreedySyncAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~GreedySyncAgent.add_negotiator
      ~GreedySyncAgent.after_join
      ~GreedySyncAgent.before_join
      ~GreedySyncAgent.before_step
      ~GreedySyncAgent.best_offer
      ~GreedySyncAgent.call
      ~GreedySyncAgent.checkpoint
      ~GreedySyncAgent.checkpoint_info
      ~GreedySyncAgent.connect_to_2021_adapter
      ~GreedySyncAgent.connect_to_oneshot_adapter
      ~GreedySyncAgent.counter_all
      ~GreedySyncAgent.create
      ~GreedySyncAgent.create_negotiator
      ~GreedySyncAgent.first_offer
      ~GreedySyncAgent.first_proposals
      ~GreedySyncAgent.from_checkpoint
      ~GreedySyncAgent.get_ami
      ~GreedySyncAgent.get_negotiator
      ~GreedySyncAgent.get_nmi
      ~GreedySyncAgent.init
      ~GreedySyncAgent.init_
      ~GreedySyncAgent.join
      ~GreedySyncAgent.kill_negotiator
      ~GreedySyncAgent.make_negotiator
      ~GreedySyncAgent.make_ufun
      ~GreedySyncAgent.on_contract_breached
      ~GreedySyncAgent.on_contract_executed
      ~GreedySyncAgent.on_leave
      ~GreedySyncAgent.on_mechanism_error
      ~GreedySyncAgent.on_negotiation_end
      ~GreedySyncAgent.on_negotiation_failure
      ~GreedySyncAgent.on_negotiation_start
      ~GreedySyncAgent.on_negotiation_success
      ~GreedySyncAgent.on_notification
      ~GreedySyncAgent.on_preferences_changed
      ~GreedySyncAgent.on_round_end
      ~GreedySyncAgent.on_round_start
      ~GreedySyncAgent.partner_agent_ids
      ~GreedySyncAgent.partner_agent_names
      ~GreedySyncAgent.partner_negotiator_ids
      ~GreedySyncAgent.partner_negotiator_names
      ~GreedySyncAgent.propose
      ~GreedySyncAgent.reset
      ~GreedySyncAgent.respond
      ~GreedySyncAgent.set_preferences
      ~GreedySyncAgent.sign_all_contracts
      ~GreedySyncAgent.spawn
      ~GreedySyncAgent.spawn_object
      ~GreedySyncAgent.step
      ~GreedySyncAgent.step_

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
