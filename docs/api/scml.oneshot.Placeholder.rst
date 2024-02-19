Placeholder
===========

.. currentmodule:: scml.oneshot

.. autoclass:: Placeholder
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Placeholder.active_negotiators
      ~Placeholder.awi
      ~Placeholder.crisp_ufun
      ~Placeholder.has_cardinal_preferences
      ~Placeholder.has_preferences
      ~Placeholder.has_ufun
      ~Placeholder.id
      ~Placeholder.internal_state
      ~Placeholder.name
      ~Placeholder.negotiators
      ~Placeholder.preferences
      ~Placeholder.prob_ufun
      ~Placeholder.reserved_outcome
      ~Placeholder.reserved_value
      ~Placeholder.running_negotiations
      ~Placeholder.short_type_name
      ~Placeholder.states
      ~Placeholder.type_name
      ~Placeholder.type_postfix
      ~Placeholder.ufun
      ~Placeholder.unsigned_contracts
      ~Placeholder.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~Placeholder.__call__
      ~Placeholder.act
      ~Placeholder.add_negotiator
      ~Placeholder.after_join
      ~Placeholder.before_join
      ~Placeholder.before_step
      ~Placeholder.call
      ~Placeholder.checkpoint
      ~Placeholder.checkpoint_info
      ~Placeholder.connect_to_2021_adapter
      ~Placeholder.connect_to_oneshot_adapter
      ~Placeholder.counter_all
      ~Placeholder.create
      ~Placeholder.create_negotiator
      ~Placeholder.decode_action
      ~Placeholder.encode_action
      ~Placeholder.encode_state
      ~Placeholder.first_offer
      ~Placeholder.first_proposals
      ~Placeholder.from_checkpoint
      ~Placeholder.get_ami
      ~Placeholder.get_negotiator
      ~Placeholder.get_nmi
      ~Placeholder.init
      ~Placeholder.init_
      ~Placeholder.join
      ~Placeholder.kill_negotiator
      ~Placeholder.make_negotiator
      ~Placeholder.make_ufun
      ~Placeholder.on_contract_breached
      ~Placeholder.on_contract_executed
      ~Placeholder.on_leave
      ~Placeholder.on_mechanism_error
      ~Placeholder.on_negotiation_end
      ~Placeholder.on_negotiation_failure
      ~Placeholder.on_negotiation_start
      ~Placeholder.on_negotiation_success
      ~Placeholder.on_notification
      ~Placeholder.on_preferences_changed
      ~Placeholder.on_round_end
      ~Placeholder.on_round_start
      ~Placeholder.partner_agent_ids
      ~Placeholder.partner_agent_names
      ~Placeholder.partner_negotiator_ids
      ~Placeholder.partner_negotiator_names
      ~Placeholder.propose
      ~Placeholder.reset
      ~Placeholder.respond
      ~Placeholder.set_preferences
      ~Placeholder.sign_all_contracts
      ~Placeholder.spawn
      ~Placeholder.spawn_object
      ~Placeholder.step
      ~Placeholder.step_

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
