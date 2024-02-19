RandomStdAgent
==============

.. currentmodule:: scml.std

.. autoclass:: RandomStdAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~RandomStdAgent.active_negotiators
      ~RandomStdAgent.awi
      ~RandomStdAgent.crisp_ufun
      ~RandomStdAgent.has_cardinal_preferences
      ~RandomStdAgent.has_preferences
      ~RandomStdAgent.has_ufun
      ~RandomStdAgent.id
      ~RandomStdAgent.internal_state
      ~RandomStdAgent.name
      ~RandomStdAgent.negotiators
      ~RandomStdAgent.preferences
      ~RandomStdAgent.prob_ufun
      ~RandomStdAgent.reserved_outcome
      ~RandomStdAgent.reserved_value
      ~RandomStdAgent.running_negotiations
      ~RandomStdAgent.short_type_name
      ~RandomStdAgent.states
      ~RandomStdAgent.type_name
      ~RandomStdAgent.type_postfix
      ~RandomStdAgent.ufun
      ~RandomStdAgent.unsigned_contracts
      ~RandomStdAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~RandomStdAgent.add_negotiator
      ~RandomStdAgent.after_join
      ~RandomStdAgent.before_join
      ~RandomStdAgent.before_step
      ~RandomStdAgent.call
      ~RandomStdAgent.checkpoint
      ~RandomStdAgent.checkpoint_info
      ~RandomStdAgent.connect_to_2021_adapter
      ~RandomStdAgent.connect_to_oneshot_adapter
      ~RandomStdAgent.create
      ~RandomStdAgent.create_negotiator
      ~RandomStdAgent.from_checkpoint
      ~RandomStdAgent.get_ami
      ~RandomStdAgent.get_negotiator
      ~RandomStdAgent.get_nmi
      ~RandomStdAgent.init
      ~RandomStdAgent.init_
      ~RandomStdAgent.join
      ~RandomStdAgent.kill_negotiator
      ~RandomStdAgent.make_negotiator
      ~RandomStdAgent.make_ufun
      ~RandomStdAgent.on_contract_breached
      ~RandomStdAgent.on_contract_executed
      ~RandomStdAgent.on_leave
      ~RandomStdAgent.on_mechanism_error
      ~RandomStdAgent.on_negotiation_end
      ~RandomStdAgent.on_negotiation_failure
      ~RandomStdAgent.on_negotiation_start
      ~RandomStdAgent.on_negotiation_success
      ~RandomStdAgent.on_notification
      ~RandomStdAgent.on_preferences_changed
      ~RandomStdAgent.on_round_end
      ~RandomStdAgent.on_round_start
      ~RandomStdAgent.partner_agent_ids
      ~RandomStdAgent.partner_agent_names
      ~RandomStdAgent.partner_negotiator_ids
      ~RandomStdAgent.partner_negotiator_names
      ~RandomStdAgent.propose
      ~RandomStdAgent.reset
      ~RandomStdAgent.respond
      ~RandomStdAgent.set_preferences
      ~RandomStdAgent.sign_all_contracts
      ~RandomStdAgent.spawn
      ~RandomStdAgent.spawn_object
      ~RandomStdAgent.step
      ~RandomStdAgent.step_

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
