RandomOneShotAgent
==================

.. currentmodule:: scml.oneshot

.. autoclass:: RandomOneShotAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~RandomOneShotAgent.active_negotiators
      ~RandomOneShotAgent.awi
      ~RandomOneShotAgent.crisp_ufun
      ~RandomOneShotAgent.has_cardinal_preferences
      ~RandomOneShotAgent.has_preferences
      ~RandomOneShotAgent.has_ufun
      ~RandomOneShotAgent.id
      ~RandomOneShotAgent.internal_state
      ~RandomOneShotAgent.name
      ~RandomOneShotAgent.negotiators
      ~RandomOneShotAgent.preferences
      ~RandomOneShotAgent.prob_ufun
      ~RandomOneShotAgent.reserved_outcome
      ~RandomOneShotAgent.reserved_value
      ~RandomOneShotAgent.running_negotiations
      ~RandomOneShotAgent.short_type_name
      ~RandomOneShotAgent.states
      ~RandomOneShotAgent.type_name
      ~RandomOneShotAgent.type_postfix
      ~RandomOneShotAgent.ufun
      ~RandomOneShotAgent.unsigned_contracts
      ~RandomOneShotAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~RandomOneShotAgent.add_negotiator
      ~RandomOneShotAgent.after_join
      ~RandomOneShotAgent.before_join
      ~RandomOneShotAgent.call
      ~RandomOneShotAgent.checkpoint
      ~RandomOneShotAgent.checkpoint_info
      ~RandomOneShotAgent.connect_to_2021_adapter
      ~RandomOneShotAgent.connect_to_oneshot_adapter
      ~RandomOneShotAgent.create
      ~RandomOneShotAgent.create_negotiator
      ~RandomOneShotAgent.from_checkpoint
      ~RandomOneShotAgent.get_ami
      ~RandomOneShotAgent.get_negotiator
      ~RandomOneShotAgent.get_nmi
      ~RandomOneShotAgent.init
      ~RandomOneShotAgent.init_
      ~RandomOneShotAgent.join
      ~RandomOneShotAgent.kill_negotiator
      ~RandomOneShotAgent.make_negotiator
      ~RandomOneShotAgent.make_ufun
      ~RandomOneShotAgent.on_contract_breached
      ~RandomOneShotAgent.on_contract_executed
      ~RandomOneShotAgent.on_leave
      ~RandomOneShotAgent.on_mechanism_error
      ~RandomOneShotAgent.on_negotiation_end
      ~RandomOneShotAgent.on_negotiation_failure
      ~RandomOneShotAgent.on_negotiation_start
      ~RandomOneShotAgent.on_negotiation_success
      ~RandomOneShotAgent.on_notification
      ~RandomOneShotAgent.on_preferences_changed
      ~RandomOneShotAgent.on_round_end
      ~RandomOneShotAgent.on_round_start
      ~RandomOneShotAgent.partner_agent_ids
      ~RandomOneShotAgent.partner_agent_names
      ~RandomOneShotAgent.partner_negotiator_ids
      ~RandomOneShotAgent.partner_negotiator_names
      ~RandomOneShotAgent.propose
      ~RandomOneShotAgent.reset
      ~RandomOneShotAgent.respond
      ~RandomOneShotAgent.set_preferences
      ~RandomOneShotAgent.sign_all_contracts
      ~RandomOneShotAgent.spawn
      ~RandomOneShotAgent.spawn_object
      ~RandomOneShotAgent.step
      ~RandomOneShotAgent.step_

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
