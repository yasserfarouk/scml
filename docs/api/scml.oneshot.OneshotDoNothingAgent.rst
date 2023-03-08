OneshotDoNothingAgent
=====================

.. currentmodule:: scml.oneshot

.. autoclass:: OneshotDoNothingAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneshotDoNothingAgent.active_negotiators
      ~OneshotDoNothingAgent.awi
      ~OneshotDoNothingAgent.crisp_ufun
      ~OneshotDoNothingAgent.has_cardinal_preferences
      ~OneshotDoNothingAgent.has_preferences
      ~OneshotDoNothingAgent.has_ufun
      ~OneshotDoNothingAgent.id
      ~OneshotDoNothingAgent.internal_state
      ~OneshotDoNothingAgent.name
      ~OneshotDoNothingAgent.negotiators
      ~OneshotDoNothingAgent.preferences
      ~OneshotDoNothingAgent.prob_ufun
      ~OneshotDoNothingAgent.reserved_outcome
      ~OneshotDoNothingAgent.reserved_value
      ~OneshotDoNothingAgent.running_negotiations
      ~OneshotDoNothingAgent.short_type_name
      ~OneshotDoNothingAgent.states
      ~OneshotDoNothingAgent.type_name
      ~OneshotDoNothingAgent.type_postfix
      ~OneshotDoNothingAgent.ufun
      ~OneshotDoNothingAgent.unsigned_contracts
      ~OneshotDoNothingAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneshotDoNothingAgent.add_negotiator
      ~OneshotDoNothingAgent.after_join
      ~OneshotDoNothingAgent.before_join
      ~OneshotDoNothingAgent.call
      ~OneshotDoNothingAgent.checkpoint
      ~OneshotDoNothingAgent.checkpoint_info
      ~OneshotDoNothingAgent.connect_to_2021_adapter
      ~OneshotDoNothingAgent.connect_to_oneshot_adapter
      ~OneshotDoNothingAgent.create
      ~OneshotDoNothingAgent.create_negotiator
      ~OneshotDoNothingAgent.from_checkpoint
      ~OneshotDoNothingAgent.get_ami
      ~OneshotDoNothingAgent.get_negotiator
      ~OneshotDoNothingAgent.get_nmi
      ~OneshotDoNothingAgent.init
      ~OneshotDoNothingAgent.init_
      ~OneshotDoNothingAgent.join
      ~OneshotDoNothingAgent.kill_negotiator
      ~OneshotDoNothingAgent.make_negotiator
      ~OneshotDoNothingAgent.make_ufun
      ~OneshotDoNothingAgent.on_contract_breached
      ~OneshotDoNothingAgent.on_contract_executed
      ~OneshotDoNothingAgent.on_leave
      ~OneshotDoNothingAgent.on_mechanism_error
      ~OneshotDoNothingAgent.on_negotiation_end
      ~OneshotDoNothingAgent.on_negotiation_failure
      ~OneshotDoNothingAgent.on_negotiation_start
      ~OneshotDoNothingAgent.on_negotiation_success
      ~OneshotDoNothingAgent.on_notification
      ~OneshotDoNothingAgent.on_preferences_changed
      ~OneshotDoNothingAgent.on_round_end
      ~OneshotDoNothingAgent.on_round_start
      ~OneshotDoNothingAgent.partner_agent_ids
      ~OneshotDoNothingAgent.partner_agent_names
      ~OneshotDoNothingAgent.partner_negotiator_ids
      ~OneshotDoNothingAgent.partner_negotiator_names
      ~OneshotDoNothingAgent.propose
      ~OneshotDoNothingAgent.reset
      ~OneshotDoNothingAgent.respond
      ~OneshotDoNothingAgent.set_preferences
      ~OneshotDoNothingAgent.sign_all_contracts
      ~OneshotDoNothingAgent.spawn
      ~OneshotDoNothingAgent.spawn_object
      ~OneshotDoNothingAgent.step
      ~OneshotDoNothingAgent.step_

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
