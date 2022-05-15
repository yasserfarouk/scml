SingleAgreementAspirationAgent
==============================

.. currentmodule:: scml.oneshot

.. autoclass:: SingleAgreementAspirationAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SingleAgreementAspirationAgent.active_negotiators
      ~SingleAgreementAspirationAgent.awi
      ~SingleAgreementAspirationAgent.crisp_ufun
      ~SingleAgreementAspirationAgent.has_cardinal_preferences
      ~SingleAgreementAspirationAgent.has_preferences
      ~SingleAgreementAspirationAgent.id
      ~SingleAgreementAspirationAgent.internal_state
      ~SingleAgreementAspirationAgent.name
      ~SingleAgreementAspirationAgent.negotiators
      ~SingleAgreementAspirationAgent.preferences
      ~SingleAgreementAspirationAgent.prob_ufun
      ~SingleAgreementAspirationAgent.reserved_outcome
      ~SingleAgreementAspirationAgent.reserved_value
      ~SingleAgreementAspirationAgent.running_negotiations
      ~SingleAgreementAspirationAgent.short_type_name
      ~SingleAgreementAspirationAgent.states
      ~SingleAgreementAspirationAgent.type_name
      ~SingleAgreementAspirationAgent.type_postfix
      ~SingleAgreementAspirationAgent.ufun
      ~SingleAgreementAspirationAgent.unsigned_contracts
      ~SingleAgreementAspirationAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~SingleAgreementAspirationAgent.add_negotiator
      ~SingleAgreementAspirationAgent.after_join
      ~SingleAgreementAspirationAgent.before_join
      ~SingleAgreementAspirationAgent.before_step
      ~SingleAgreementAspirationAgent.call
      ~SingleAgreementAspirationAgent.checkpoint
      ~SingleAgreementAspirationAgent.checkpoint_info
      ~SingleAgreementAspirationAgent.choose_agents
      ~SingleAgreementAspirationAgent.connect_to_2021_adapter
      ~SingleAgreementAspirationAgent.connect_to_oneshot_adapter
      ~SingleAgreementAspirationAgent.counter_all
      ~SingleAgreementAspirationAgent.create
      ~SingleAgreementAspirationAgent.create_negotiator
      ~SingleAgreementAspirationAgent.first_offer
      ~SingleAgreementAspirationAgent.first_proposals
      ~SingleAgreementAspirationAgent.from_checkpoint
      ~SingleAgreementAspirationAgent.get_ami
      ~SingleAgreementAspirationAgent.get_negotiator
      ~SingleAgreementAspirationAgent.get_nmi
      ~SingleAgreementAspirationAgent.init
      ~SingleAgreementAspirationAgent.init_
      ~SingleAgreementAspirationAgent.join
      ~SingleAgreementAspirationAgent.kill_negotiator
      ~SingleAgreementAspirationAgent.make_negotiator
      ~SingleAgreementAspirationAgent.make_ufun
      ~SingleAgreementAspirationAgent.on_contract_breached
      ~SingleAgreementAspirationAgent.on_contract_executed
      ~SingleAgreementAspirationAgent.on_leave
      ~SingleAgreementAspirationAgent.on_mechanism_error
      ~SingleAgreementAspirationAgent.on_negotiation_end
      ~SingleAgreementAspirationAgent.on_negotiation_failure
      ~SingleAgreementAspirationAgent.on_negotiation_start
      ~SingleAgreementAspirationAgent.on_negotiation_success
      ~SingleAgreementAspirationAgent.on_notification
      ~SingleAgreementAspirationAgent.on_preferences_changed
      ~SingleAgreementAspirationAgent.on_round_end
      ~SingleAgreementAspirationAgent.on_round_start
      ~SingleAgreementAspirationAgent.partner_agent_ids
      ~SingleAgreementAspirationAgent.partner_agent_names
      ~SingleAgreementAspirationAgent.partner_negotiator_ids
      ~SingleAgreementAspirationAgent.partner_negotiator_names
      ~SingleAgreementAspirationAgent.propose
      ~SingleAgreementAspirationAgent.reset
      ~SingleAgreementAspirationAgent.respond
      ~SingleAgreementAspirationAgent.set_preferences
      ~SingleAgreementAspirationAgent.sign_all_contracts
      ~SingleAgreementAspirationAgent.spawn
      ~SingleAgreementAspirationAgent.spawn_object
      ~SingleAgreementAspirationAgent.step
      ~SingleAgreementAspirationAgent.step_

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
   .. automethod:: before_step
   .. automethod:: call
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: choose_agents
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
