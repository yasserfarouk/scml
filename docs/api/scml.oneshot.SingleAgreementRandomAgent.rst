SingleAgreementRandomAgent
==========================

.. currentmodule:: scml.oneshot

.. autoclass:: SingleAgreementRandomAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SingleAgreementRandomAgent.active_negotiators
      ~SingleAgreementRandomAgent.awi
      ~SingleAgreementRandomAgent.crisp_ufun
      ~SingleAgreementRandomAgent.has_cardinal_preferences
      ~SingleAgreementRandomAgent.has_preferences
      ~SingleAgreementRandomAgent.has_ufun
      ~SingleAgreementRandomAgent.id
      ~SingleAgreementRandomAgent.internal_state
      ~SingleAgreementRandomAgent.name
      ~SingleAgreementRandomAgent.negotiators
      ~SingleAgreementRandomAgent.preferences
      ~SingleAgreementRandomAgent.prob_ufun
      ~SingleAgreementRandomAgent.reserved_outcome
      ~SingleAgreementRandomAgent.reserved_value
      ~SingleAgreementRandomAgent.running_negotiations
      ~SingleAgreementRandomAgent.short_type_name
      ~SingleAgreementRandomAgent.states
      ~SingleAgreementRandomAgent.type_name
      ~SingleAgreementRandomAgent.type_postfix
      ~SingleAgreementRandomAgent.ufun
      ~SingleAgreementRandomAgent.unsigned_contracts
      ~SingleAgreementRandomAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~SingleAgreementRandomAgent.add_negotiator
      ~SingleAgreementRandomAgent.after_join
      ~SingleAgreementRandomAgent.before_join
      ~SingleAgreementRandomAgent.before_step
      ~SingleAgreementRandomAgent.best_offer
      ~SingleAgreementRandomAgent.best_outcome
      ~SingleAgreementRandomAgent.call
      ~SingleAgreementRandomAgent.checkpoint
      ~SingleAgreementRandomAgent.checkpoint_info
      ~SingleAgreementRandomAgent.connect_to_2021_adapter
      ~SingleAgreementRandomAgent.connect_to_oneshot_adapter
      ~SingleAgreementRandomAgent.counter_all
      ~SingleAgreementRandomAgent.create
      ~SingleAgreementRandomAgent.create_negotiator
      ~SingleAgreementRandomAgent.first_offer
      ~SingleAgreementRandomAgent.first_proposals
      ~SingleAgreementRandomAgent.from_checkpoint
      ~SingleAgreementRandomAgent.get_ami
      ~SingleAgreementRandomAgent.get_negotiator
      ~SingleAgreementRandomAgent.get_nmi
      ~SingleAgreementRandomAgent.init
      ~SingleAgreementRandomAgent.init_
      ~SingleAgreementRandomAgent.is_acceptable
      ~SingleAgreementRandomAgent.is_better
      ~SingleAgreementRandomAgent.join
      ~SingleAgreementRandomAgent.kill_negotiator
      ~SingleAgreementRandomAgent.make_negotiator
      ~SingleAgreementRandomAgent.make_offer
      ~SingleAgreementRandomAgent.make_ufun
      ~SingleAgreementRandomAgent.on_contract_breached
      ~SingleAgreementRandomAgent.on_contract_executed
      ~SingleAgreementRandomAgent.on_leave
      ~SingleAgreementRandomAgent.on_mechanism_error
      ~SingleAgreementRandomAgent.on_negotiation_end
      ~SingleAgreementRandomAgent.on_negotiation_failure
      ~SingleAgreementRandomAgent.on_negotiation_start
      ~SingleAgreementRandomAgent.on_negotiation_success
      ~SingleAgreementRandomAgent.on_notification
      ~SingleAgreementRandomAgent.on_preferences_changed
      ~SingleAgreementRandomAgent.on_round_end
      ~SingleAgreementRandomAgent.on_round_start
      ~SingleAgreementRandomAgent.partner_agent_ids
      ~SingleAgreementRandomAgent.partner_agent_names
      ~SingleAgreementRandomAgent.partner_negotiator_ids
      ~SingleAgreementRandomAgent.partner_negotiator_names
      ~SingleAgreementRandomAgent.propose
      ~SingleAgreementRandomAgent.reset
      ~SingleAgreementRandomAgent.respond
      ~SingleAgreementRandomAgent.response_to_best_offer
      ~SingleAgreementRandomAgent.set_preferences
      ~SingleAgreementRandomAgent.sign_all_contracts
      ~SingleAgreementRandomAgent.spawn
      ~SingleAgreementRandomAgent.spawn_object
      ~SingleAgreementRandomAgent.step
      ~SingleAgreementRandomAgent.step_

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
   .. automethod:: best_outcome
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
   .. automethod:: is_acceptable
   .. automethod:: is_better
   .. automethod:: join
   .. automethod:: kill_negotiator
   .. automethod:: make_negotiator
   .. automethod:: make_offer
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
   .. automethod:: response_to_best_offer
   .. automethod:: set_preferences
   .. automethod:: sign_all_contracts
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: step
   .. automethod:: step_
