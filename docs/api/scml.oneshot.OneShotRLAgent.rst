OneShotRLAgent
==============

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotRLAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotRLAgent.active_negotiators
      ~OneShotRLAgent.awi
      ~OneShotRLAgent.crisp_ufun
      ~OneShotRLAgent.has_cardinal_preferences
      ~OneShotRLAgent.has_preferences
      ~OneShotRLAgent.has_ufun
      ~OneShotRLAgent.id
      ~OneShotRLAgent.internal_state
      ~OneShotRLAgent.name
      ~OneShotRLAgent.negotiators
      ~OneShotRLAgent.preferences
      ~OneShotRLAgent.prob_ufun
      ~OneShotRLAgent.reserved_outcome
      ~OneShotRLAgent.reserved_value
      ~OneShotRLAgent.running_negotiations
      ~OneShotRLAgent.short_type_name
      ~OneShotRLAgent.states
      ~OneShotRLAgent.type_name
      ~OneShotRLAgent.type_postfix
      ~OneShotRLAgent.ufun
      ~OneShotRLAgent.unsigned_contracts
      ~OneShotRLAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotRLAgent.__call__
      ~OneShotRLAgent.act
      ~OneShotRLAgent.add_negotiator
      ~OneShotRLAgent.after_join
      ~OneShotRLAgent.before_join
      ~OneShotRLAgent.before_step
      ~OneShotRLAgent.call
      ~OneShotRLAgent.checkpoint
      ~OneShotRLAgent.checkpoint_info
      ~OneShotRLAgent.connect_to_2021_adapter
      ~OneShotRLAgent.connect_to_oneshot_adapter
      ~OneShotRLAgent.context_switch
      ~OneShotRLAgent.counter_all
      ~OneShotRLAgent.create
      ~OneShotRLAgent.create_negotiator
      ~OneShotRLAgent.decode_action
      ~OneShotRLAgent.encode_action
      ~OneShotRLAgent.encode_state
      ~OneShotRLAgent.first_offer
      ~OneShotRLAgent.first_proposals
      ~OneShotRLAgent.from_checkpoint
      ~OneShotRLAgent.get_ami
      ~OneShotRLAgent.get_negotiator
      ~OneShotRLAgent.get_nmi
      ~OneShotRLAgent.has_no_valid_model
      ~OneShotRLAgent.init
      ~OneShotRLAgent.init_
      ~OneShotRLAgent.join
      ~OneShotRLAgent.kill_negotiator
      ~OneShotRLAgent.make_negotiator
      ~OneShotRLAgent.make_ufun
      ~OneShotRLAgent.on_contract_breached
      ~OneShotRLAgent.on_contract_executed
      ~OneShotRLAgent.on_leave
      ~OneShotRLAgent.on_mechanism_error
      ~OneShotRLAgent.on_negotiation_end
      ~OneShotRLAgent.on_negotiation_failure
      ~OneShotRLAgent.on_negotiation_start
      ~OneShotRLAgent.on_negotiation_success
      ~OneShotRLAgent.on_notification
      ~OneShotRLAgent.on_preferences_changed
      ~OneShotRLAgent.on_round_end
      ~OneShotRLAgent.on_round_start
      ~OneShotRLAgent.partner_agent_ids
      ~OneShotRLAgent.partner_agent_names
      ~OneShotRLAgent.partner_negotiator_ids
      ~OneShotRLAgent.partner_negotiator_names
      ~OneShotRLAgent.propose
      ~OneShotRLAgent.reset
      ~OneShotRLAgent.respond
      ~OneShotRLAgent.set_preferences
      ~OneShotRLAgent.setup_fallback
      ~OneShotRLAgent.sign_all_contracts
      ~OneShotRLAgent.spawn
      ~OneShotRLAgent.spawn_object
      ~OneShotRLAgent.step
      ~OneShotRLAgent.step_

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

   .. automethod:: __call__
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
   .. automethod:: context_switch
   .. automethod:: counter_all
   .. automethod:: create
   .. automethod:: create_negotiator
   .. automethod:: decode_action
   .. automethod:: encode_action
   .. automethod:: encode_state
   .. automethod:: first_offer
   .. automethod:: first_proposals
   .. automethod:: from_checkpoint
   .. automethod:: get_ami
   .. automethod:: get_negotiator
   .. automethod:: get_nmi
   .. automethod:: has_no_valid_model
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
   .. automethod:: setup_fallback
   .. automethod:: sign_all_contracts
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: step
   .. automethod:: step_
