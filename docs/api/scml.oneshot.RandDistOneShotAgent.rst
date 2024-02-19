RandDistOneShotAgent
====================

.. currentmodule:: scml.oneshot

.. autoclass:: RandDistOneShotAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~RandDistOneShotAgent.active_negotiators
      ~RandDistOneShotAgent.awi
      ~RandDistOneShotAgent.crisp_ufun
      ~RandDistOneShotAgent.has_cardinal_preferences
      ~RandDistOneShotAgent.has_preferences
      ~RandDistOneShotAgent.has_ufun
      ~RandDistOneShotAgent.id
      ~RandDistOneShotAgent.internal_state
      ~RandDistOneShotAgent.name
      ~RandDistOneShotAgent.negotiators
      ~RandDistOneShotAgent.preferences
      ~RandDistOneShotAgent.prob_ufun
      ~RandDistOneShotAgent.reserved_outcome
      ~RandDistOneShotAgent.reserved_value
      ~RandDistOneShotAgent.running_negotiations
      ~RandDistOneShotAgent.short_type_name
      ~RandDistOneShotAgent.states
      ~RandDistOneShotAgent.type_name
      ~RandDistOneShotAgent.type_postfix
      ~RandDistOneShotAgent.ufun
      ~RandDistOneShotAgent.unsigned_contracts
      ~RandDistOneShotAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~RandDistOneShotAgent.add_negotiator
      ~RandDistOneShotAgent.after_join
      ~RandDistOneShotAgent.before_join
      ~RandDistOneShotAgent.before_step
      ~RandDistOneShotAgent.call
      ~RandDistOneShotAgent.checkpoint
      ~RandDistOneShotAgent.checkpoint_info
      ~RandDistOneShotAgent.connect_to_2021_adapter
      ~RandDistOneShotAgent.connect_to_oneshot_adapter
      ~RandDistOneShotAgent.counter_all
      ~RandDistOneShotAgent.create
      ~RandDistOneShotAgent.create_negotiator
      ~RandDistOneShotAgent.distribute_needs
      ~RandDistOneShotAgent.first_offer
      ~RandDistOneShotAgent.first_proposals
      ~RandDistOneShotAgent.from_checkpoint
      ~RandDistOneShotAgent.get_ami
      ~RandDistOneShotAgent.get_negotiator
      ~RandDistOneShotAgent.get_nmi
      ~RandDistOneShotAgent.init
      ~RandDistOneShotAgent.init_
      ~RandDistOneShotAgent.join
      ~RandDistOneShotAgent.kill_negotiator
      ~RandDistOneShotAgent.make_negotiator
      ~RandDistOneShotAgent.make_ufun
      ~RandDistOneShotAgent.on_contract_breached
      ~RandDistOneShotAgent.on_contract_executed
      ~RandDistOneShotAgent.on_leave
      ~RandDistOneShotAgent.on_mechanism_error
      ~RandDistOneShotAgent.on_negotiation_end
      ~RandDistOneShotAgent.on_negotiation_failure
      ~RandDistOneShotAgent.on_negotiation_start
      ~RandDistOneShotAgent.on_negotiation_success
      ~RandDistOneShotAgent.on_notification
      ~RandDistOneShotAgent.on_preferences_changed
      ~RandDistOneShotAgent.on_round_end
      ~RandDistOneShotAgent.on_round_start
      ~RandDistOneShotAgent.partner_agent_ids
      ~RandDistOneShotAgent.partner_agent_names
      ~RandDistOneShotAgent.partner_negotiator_ids
      ~RandDistOneShotAgent.partner_negotiator_names
      ~RandDistOneShotAgent.propose
      ~RandDistOneShotAgent.reset
      ~RandDistOneShotAgent.respond
      ~RandDistOneShotAgent.set_preferences
      ~RandDistOneShotAgent.sign_all_contracts
      ~RandDistOneShotAgent.spawn
      ~RandDistOneShotAgent.spawn_object
      ~RandDistOneShotAgent.step
      ~RandDistOneShotAgent.step_

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
   .. automethod:: counter_all
   .. automethod:: create
   .. automethod:: create_negotiator
   .. automethod:: distribute_needs
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
