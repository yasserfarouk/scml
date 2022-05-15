OneShotAgent
============

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotAgent.active_negotiators
      ~OneShotAgent.awi
      ~OneShotAgent.crisp_ufun
      ~OneShotAgent.has_cardinal_preferences
      ~OneShotAgent.has_preferences
      ~OneShotAgent.id
      ~OneShotAgent.internal_state
      ~OneShotAgent.name
      ~OneShotAgent.negotiators
      ~OneShotAgent.preferences
      ~OneShotAgent.prob_ufun
      ~OneShotAgent.reserved_outcome
      ~OneShotAgent.reserved_value
      ~OneShotAgent.running_negotiations
      ~OneShotAgent.short_type_name
      ~OneShotAgent.states
      ~OneShotAgent.type_name
      ~OneShotAgent.type_postfix
      ~OneShotAgent.ufun
      ~OneShotAgent.unsigned_contracts
      ~OneShotAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotAgent.add_negotiator
      ~OneShotAgent.after_join
      ~OneShotAgent.before_join
      ~OneShotAgent.call
      ~OneShotAgent.checkpoint
      ~OneShotAgent.checkpoint_info
      ~OneShotAgent.connect_to_2021_adapter
      ~OneShotAgent.connect_to_oneshot_adapter
      ~OneShotAgent.create
      ~OneShotAgent.create_negotiator
      ~OneShotAgent.from_checkpoint
      ~OneShotAgent.get_ami
      ~OneShotAgent.get_negotiator
      ~OneShotAgent.get_nmi
      ~OneShotAgent.init
      ~OneShotAgent.init_
      ~OneShotAgent.join
      ~OneShotAgent.kill_negotiator
      ~OneShotAgent.make_negotiator
      ~OneShotAgent.make_ufun
      ~OneShotAgent.on_contract_breached
      ~OneShotAgent.on_contract_executed
      ~OneShotAgent.on_leave
      ~OneShotAgent.on_mechanism_error
      ~OneShotAgent.on_negotiation_end
      ~OneShotAgent.on_negotiation_failure
      ~OneShotAgent.on_negotiation_start
      ~OneShotAgent.on_negotiation_success
      ~OneShotAgent.on_notification
      ~OneShotAgent.on_preferences_changed
      ~OneShotAgent.on_round_end
      ~OneShotAgent.on_round_start
      ~OneShotAgent.partner_agent_ids
      ~OneShotAgent.partner_agent_names
      ~OneShotAgent.partner_negotiator_ids
      ~OneShotAgent.partner_negotiator_names
      ~OneShotAgent.propose
      ~OneShotAgent.reset
      ~OneShotAgent.respond
      ~OneShotAgent.set_preferences
      ~OneShotAgent.sign_all_contracts
      ~OneShotAgent.spawn
      ~OneShotAgent.spawn_object
      ~OneShotAgent.step
      ~OneShotAgent.step_

   .. rubric:: Attributes Documentation

   .. autoattribute:: active_negotiators
   .. autoattribute:: awi
   .. autoattribute:: crisp_ufun
   .. autoattribute:: has_cardinal_preferences
   .. autoattribute:: has_preferences
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
