GreedySingleAgreementAgent
==========================

.. currentmodule:: scml.oneshot

.. autoclass:: GreedySingleAgreementAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~GreedySingleAgreementAgent.active_negotiators
      ~GreedySingleAgreementAgent.awi
      ~GreedySingleAgreementAgent.crisp_ufun
      ~GreedySingleAgreementAgent.has_cardinal_preferences
      ~GreedySingleAgreementAgent.has_preferences
      ~GreedySingleAgreementAgent.id
      ~GreedySingleAgreementAgent.internal_state
      ~GreedySingleAgreementAgent.name
      ~GreedySingleAgreementAgent.negotiators
      ~GreedySingleAgreementAgent.preferences
      ~GreedySingleAgreementAgent.prob_ufun
      ~GreedySingleAgreementAgent.reserved_outcome
      ~GreedySingleAgreementAgent.reserved_value
      ~GreedySingleAgreementAgent.running_negotiations
      ~GreedySingleAgreementAgent.short_type_name
      ~GreedySingleAgreementAgent.states
      ~GreedySingleAgreementAgent.type_name
      ~GreedySingleAgreementAgent.type_postfix
      ~GreedySingleAgreementAgent.ufun
      ~GreedySingleAgreementAgent.unsigned_contracts
      ~GreedySingleAgreementAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~GreedySingleAgreementAgent.add_negotiator
      ~GreedySingleAgreementAgent.after_join
      ~GreedySingleAgreementAgent.before_join
      ~GreedySingleAgreementAgent.before_step
      ~GreedySingleAgreementAgent.best_offer
      ~GreedySingleAgreementAgent.best_outcome
      ~GreedySingleAgreementAgent.call
      ~GreedySingleAgreementAgent.checkpoint
      ~GreedySingleAgreementAgent.checkpoint_info
      ~GreedySingleAgreementAgent.connect_to_2021_adapter
      ~GreedySingleAgreementAgent.connect_to_oneshot_adapter
      ~GreedySingleAgreementAgent.counter_all
      ~GreedySingleAgreementAgent.create
      ~GreedySingleAgreementAgent.create_negotiator
      ~GreedySingleAgreementAgent.first_offer
      ~GreedySingleAgreementAgent.first_proposals
      ~GreedySingleAgreementAgent.from_checkpoint
      ~GreedySingleAgreementAgent.get_ami
      ~GreedySingleAgreementAgent.get_negotiator
      ~GreedySingleAgreementAgent.get_nmi
      ~GreedySingleAgreementAgent.init
      ~GreedySingleAgreementAgent.init_
      ~GreedySingleAgreementAgent.is_acceptable
      ~GreedySingleAgreementAgent.is_better
      ~GreedySingleAgreementAgent.join
      ~GreedySingleAgreementAgent.kill_negotiator
      ~GreedySingleAgreementAgent.make_negotiator
      ~GreedySingleAgreementAgent.make_offer
      ~GreedySingleAgreementAgent.make_ufun
      ~GreedySingleAgreementAgent.on_contract_breached
      ~GreedySingleAgreementAgent.on_contract_executed
      ~GreedySingleAgreementAgent.on_leave
      ~GreedySingleAgreementAgent.on_mechanism_error
      ~GreedySingleAgreementAgent.on_negotiation_end
      ~GreedySingleAgreementAgent.on_negotiation_failure
      ~GreedySingleAgreementAgent.on_negotiation_start
      ~GreedySingleAgreementAgent.on_negotiation_success
      ~GreedySingleAgreementAgent.on_notification
      ~GreedySingleAgreementAgent.on_preferences_changed
      ~GreedySingleAgreementAgent.on_round_end
      ~GreedySingleAgreementAgent.on_round_start
      ~GreedySingleAgreementAgent.partner_agent_ids
      ~GreedySingleAgreementAgent.partner_agent_names
      ~GreedySingleAgreementAgent.partner_negotiator_ids
      ~GreedySingleAgreementAgent.partner_negotiator_names
      ~GreedySingleAgreementAgent.propose
      ~GreedySingleAgreementAgent.reset
      ~GreedySingleAgreementAgent.respond
      ~GreedySingleAgreementAgent.response_to_best_offer
      ~GreedySingleAgreementAgent.set_preferences
      ~GreedySingleAgreementAgent.sign_all_contracts
      ~GreedySingleAgreementAgent.spawn
      ~GreedySingleAgreementAgent.spawn_object
      ~GreedySingleAgreementAgent.step
      ~GreedySingleAgreementAgent.step_

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
