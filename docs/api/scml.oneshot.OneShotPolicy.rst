OneShotPolicy
=============

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotPolicy
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotPolicy.active_negotiators
      ~OneShotPolicy.awi
      ~OneShotPolicy.crisp_ufun
      ~OneShotPolicy.has_cardinal_preferences
      ~OneShotPolicy.has_preferences
      ~OneShotPolicy.has_ufun
      ~OneShotPolicy.id
      ~OneShotPolicy.internal_state
      ~OneShotPolicy.name
      ~OneShotPolicy.negotiators
      ~OneShotPolicy.preferences
      ~OneShotPolicy.prob_ufun
      ~OneShotPolicy.reserved_outcome
      ~OneShotPolicy.reserved_value
      ~OneShotPolicy.running_negotiations
      ~OneShotPolicy.short_type_name
      ~OneShotPolicy.states
      ~OneShotPolicy.type_name
      ~OneShotPolicy.type_postfix
      ~OneShotPolicy.ufun
      ~OneShotPolicy.unsigned_contracts
      ~OneShotPolicy.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotPolicy.act
      ~OneShotPolicy.add_negotiator
      ~OneShotPolicy.after_join
      ~OneShotPolicy.before_join
      ~OneShotPolicy.before_step
      ~OneShotPolicy.call
      ~OneShotPolicy.checkpoint
      ~OneShotPolicy.checkpoint_info
      ~OneShotPolicy.connect_to_2021_adapter
      ~OneShotPolicy.connect_to_oneshot_adapter
      ~OneShotPolicy.counter_all
      ~OneShotPolicy.create
      ~OneShotPolicy.create_negotiator
      ~OneShotPolicy.decode_action
      ~OneShotPolicy.encode_state
      ~OneShotPolicy.first_offer
      ~OneShotPolicy.first_proposals
      ~OneShotPolicy.from_checkpoint
      ~OneShotPolicy.get_ami
      ~OneShotPolicy.get_negotiator
      ~OneShotPolicy.get_nmi
      ~OneShotPolicy.init
      ~OneShotPolicy.init_
      ~OneShotPolicy.join
      ~OneShotPolicy.kill_negotiator
      ~OneShotPolicy.make_negotiator
      ~OneShotPolicy.make_ufun
      ~OneShotPolicy.on_contract_breached
      ~OneShotPolicy.on_contract_executed
      ~OneShotPolicy.on_leave
      ~OneShotPolicy.on_mechanism_error
      ~OneShotPolicy.on_negotiation_end
      ~OneShotPolicy.on_negotiation_failure
      ~OneShotPolicy.on_negotiation_start
      ~OneShotPolicy.on_negotiation_success
      ~OneShotPolicy.on_notification
      ~OneShotPolicy.on_preferences_changed
      ~OneShotPolicy.on_round_end
      ~OneShotPolicy.on_round_start
      ~OneShotPolicy.partner_agent_ids
      ~OneShotPolicy.partner_agent_names
      ~OneShotPolicy.partner_negotiator_ids
      ~OneShotPolicy.partner_negotiator_names
      ~OneShotPolicy.propose
      ~OneShotPolicy.reset
      ~OneShotPolicy.respond
      ~OneShotPolicy.set_preferences
      ~OneShotPolicy.sign_all_contracts
      ~OneShotPolicy.spawn
      ~OneShotPolicy.spawn_object
      ~OneShotPolicy.step
      ~OneShotPolicy.step_

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

   .. automethod:: act
   .. automethod:: add_negotiator
   .. automethod:: after_join
   .. automethod:: before_join
   .. automethod:: before_step
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: connect_to_2021_adapter
   .. automethod:: connect_to_oneshot_adapter
   .. automethod:: counter_all
   .. automethod:: create
   .. automethod:: create_negotiator
   .. automethod:: decode_action
   .. automethod:: encode_state
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
