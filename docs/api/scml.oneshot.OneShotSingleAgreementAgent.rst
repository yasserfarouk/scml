OneShotSingleAgreementAgent
===========================

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotSingleAgreementAgent
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotSingleAgreementAgent.active_negotiators
      ~OneShotSingleAgreementAgent.awi
      ~OneShotSingleAgreementAgent.crisp_ufun
      ~OneShotSingleAgreementAgent.has_cardinal_preferences
      ~OneShotSingleAgreementAgent.has_preferences
      ~OneShotSingleAgreementAgent.id
      ~OneShotSingleAgreementAgent.internal_state
      ~OneShotSingleAgreementAgent.name
      ~OneShotSingleAgreementAgent.negotiators
      ~OneShotSingleAgreementAgent.preferences
      ~OneShotSingleAgreementAgent.prob_ufun
      ~OneShotSingleAgreementAgent.reserved_outcome
      ~OneShotSingleAgreementAgent.reserved_value
      ~OneShotSingleAgreementAgent.running_negotiations
      ~OneShotSingleAgreementAgent.short_type_name
      ~OneShotSingleAgreementAgent.states
      ~OneShotSingleAgreementAgent.type_name
      ~OneShotSingleAgreementAgent.type_postfix
      ~OneShotSingleAgreementAgent.ufun
      ~OneShotSingleAgreementAgent.unsigned_contracts
      ~OneShotSingleAgreementAgent.uuid

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotSingleAgreementAgent.add_negotiator
      ~OneShotSingleAgreementAgent.after_join
      ~OneShotSingleAgreementAgent.before_join
      ~OneShotSingleAgreementAgent.best_offer
      ~OneShotSingleAgreementAgent.best_outcome
      ~OneShotSingleAgreementAgent.call
      ~OneShotSingleAgreementAgent.checkpoint
      ~OneShotSingleAgreementAgent.checkpoint_info
      ~OneShotSingleAgreementAgent.connect_to_2021_adapter
      ~OneShotSingleAgreementAgent.connect_to_oneshot_adapter
      ~OneShotSingleAgreementAgent.counter_all
      ~OneShotSingleAgreementAgent.create
      ~OneShotSingleAgreementAgent.create_negotiator
      ~OneShotSingleAgreementAgent.first_offer
      ~OneShotSingleAgreementAgent.first_proposals
      ~OneShotSingleAgreementAgent.from_checkpoint
      ~OneShotSingleAgreementAgent.get_ami
      ~OneShotSingleAgreementAgent.get_negotiator
      ~OneShotSingleAgreementAgent.get_nmi
      ~OneShotSingleAgreementAgent.init
      ~OneShotSingleAgreementAgent.init_
      ~OneShotSingleAgreementAgent.is_acceptable
      ~OneShotSingleAgreementAgent.is_better
      ~OneShotSingleAgreementAgent.join
      ~OneShotSingleAgreementAgent.kill_negotiator
      ~OneShotSingleAgreementAgent.make_negotiator
      ~OneShotSingleAgreementAgent.make_offer
      ~OneShotSingleAgreementAgent.make_ufun
      ~OneShotSingleAgreementAgent.on_contract_breached
      ~OneShotSingleAgreementAgent.on_contract_executed
      ~OneShotSingleAgreementAgent.on_leave
      ~OneShotSingleAgreementAgent.on_mechanism_error
      ~OneShotSingleAgreementAgent.on_negotiation_end
      ~OneShotSingleAgreementAgent.on_negotiation_failure
      ~OneShotSingleAgreementAgent.on_negotiation_start
      ~OneShotSingleAgreementAgent.on_negotiation_success
      ~OneShotSingleAgreementAgent.on_notification
      ~OneShotSingleAgreementAgent.on_preferences_changed
      ~OneShotSingleAgreementAgent.on_round_end
      ~OneShotSingleAgreementAgent.on_round_start
      ~OneShotSingleAgreementAgent.partner_agent_ids
      ~OneShotSingleAgreementAgent.partner_agent_names
      ~OneShotSingleAgreementAgent.partner_negotiator_ids
      ~OneShotSingleAgreementAgent.partner_negotiator_names
      ~OneShotSingleAgreementAgent.propose
      ~OneShotSingleAgreementAgent.reset
      ~OneShotSingleAgreementAgent.respond
      ~OneShotSingleAgreementAgent.response_to_best_offer
      ~OneShotSingleAgreementAgent.set_preferences
      ~OneShotSingleAgreementAgent.sign_all_contracts
      ~OneShotSingleAgreementAgent.spawn
      ~OneShotSingleAgreementAgent.spawn_object
      ~OneShotSingleAgreementAgent.step
      ~OneShotSingleAgreementAgent.step_

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
