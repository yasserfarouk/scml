OneShotUFun
===========

.. currentmodule:: scml.oneshot

.. autoclass:: OneShotUFun
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OneShotUFun.base_type
      ~OneShotUFun.best_option
      ~OneShotUFun.id
      ~OneShotUFun.max_utility
      ~OneShotUFun.min_utility
      ~OneShotUFun.name
      ~OneShotUFun.owner
      ~OneShotUFun.reserved_distribution
      ~OneShotUFun.short_type_name
      ~OneShotUFun.type
      ~OneShotUFun.type_name
      ~OneShotUFun.uuid
      ~OneShotUFun.worst_option

   .. rubric:: Methods Summary

   .. autosummary::

      ~OneShotUFun.__call__
      ~OneShotUFun.argrank
      ~OneShotUFun.argrank_with_weights
      ~OneShotUFun.best
      ~OneShotUFun.breach_level
      ~OneShotUFun.changes
      ~OneShotUFun.checkpoint
      ~OneShotUFun.checkpoint_info
      ~OneShotUFun.create
      ~OneShotUFun.difference
      ~OneShotUFun.difference_prob
      ~OneShotUFun.eu
      ~OneShotUFun.eval
      ~OneShotUFun.eval_normalized
      ~OneShotUFun.extreme_outcomes
      ~OneShotUFun.find_limit
      ~OneShotUFun.find_limit_brute_force
      ~OneShotUFun.from_aggregates
      ~OneShotUFun.from_checkpoint
      ~OneShotUFun.from_contracts
      ~OneShotUFun.from_dict
      ~OneShotUFun.from_genius
      ~OneShotUFun.from_geniusweb
      ~OneShotUFun.from_geniusweb_json_str
      ~OneShotUFun.from_offers
      ~OneShotUFun.from_xml_str
      ~OneShotUFun.generate_bilateral
      ~OneShotUFun.generate_random
      ~OneShotUFun.generate_random_bilateral
      ~OneShotUFun.invert
      ~OneShotUFun.is_better
      ~OneShotUFun.is_breach
      ~OneShotUFun.is_equivalent
      ~OneShotUFun.is_not_better
      ~OneShotUFun.is_not_worse
      ~OneShotUFun.is_session_dependent
      ~OneShotUFun.is_state_dependent
      ~OneShotUFun.is_stationary
      ~OneShotUFun.is_volatile
      ~OneShotUFun.is_worse
      ~OneShotUFun.max
      ~OneShotUFun.min
      ~OneShotUFun.minmax
      ~OneShotUFun.normalize
      ~OneShotUFun.normalize_for
      ~OneShotUFun.ok_to_buy_at
      ~OneShotUFun.ok_to_sell_at
      ~OneShotUFun.outcome_as_tuple
      ~OneShotUFun.rank
      ~OneShotUFun.rank_with_weights
      ~OneShotUFun.register_sale
      ~OneShotUFun.register_sale_failure
      ~OneShotUFun.register_supply
      ~OneShotUFun.register_supply_failure
      ~OneShotUFun.reset_changes
      ~OneShotUFun.sample_outcome_with_utility
      ~OneShotUFun.scale_by
      ~OneShotUFun.scale_max
      ~OneShotUFun.scale_max_for
      ~OneShotUFun.scale_min
      ~OneShotUFun.scale_min_for
      ~OneShotUFun.shift_by
      ~OneShotUFun.shift_max_for
      ~OneShotUFun.shift_min_for
      ~OneShotUFun.spawn
      ~OneShotUFun.spawn_object
      ~OneShotUFun.to_crisp
      ~OneShotUFun.to_dict
      ~OneShotUFun.to_genius
      ~OneShotUFun.to_prob
      ~OneShotUFun.to_stationary
      ~OneShotUFun.to_xml_str
      ~OneShotUFun.utility_range
      ~OneShotUFun.worst
      ~OneShotUFun.xml

   .. rubric:: Attributes Documentation

   .. autoattribute:: base_type
   .. autoattribute:: best_option
   .. autoattribute:: id
   .. autoattribute:: max_utility
   .. autoattribute:: min_utility
   .. autoattribute:: name
   .. autoattribute:: owner
   .. autoattribute:: reserved_distribution
   .. autoattribute:: short_type_name
   .. autoattribute:: type
   .. autoattribute:: type_name
   .. autoattribute:: uuid
   .. autoattribute:: worst_option

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: argrank
   .. automethod:: argrank_with_weights
   .. automethod:: best
   .. automethod:: breach_level
   .. automethod:: changes
   .. automethod:: checkpoint
   .. automethod:: checkpoint_info
   .. automethod:: create
   .. automethod:: difference
   .. automethod:: difference_prob
   .. automethod:: eu
   .. automethod:: eval
   .. automethod:: eval_normalized
   .. automethod:: extreme_outcomes
   .. automethod:: find_limit
   .. automethod:: find_limit_brute_force
   .. automethod:: from_aggregates
   .. automethod:: from_checkpoint
   .. automethod:: from_contracts
   .. automethod:: from_dict
   .. automethod:: from_genius
   .. automethod:: from_geniusweb
   .. automethod:: from_geniusweb_json_str
   .. automethod:: from_offers
   .. automethod:: from_xml_str
   .. automethod:: generate_bilateral
   .. automethod:: generate_random
   .. automethod:: generate_random_bilateral
   .. automethod:: invert
   .. automethod:: is_better
   .. automethod:: is_breach
   .. automethod:: is_equivalent
   .. automethod:: is_not_better
   .. automethod:: is_not_worse
   .. automethod:: is_session_dependent
   .. automethod:: is_state_dependent
   .. automethod:: is_stationary
   .. automethod:: is_volatile
   .. automethod:: is_worse
   .. automethod:: max
   .. automethod:: min
   .. automethod:: minmax
   .. automethod:: normalize
   .. automethod:: normalize_for
   .. automethod:: ok_to_buy_at
   .. automethod:: ok_to_sell_at
   .. automethod:: outcome_as_tuple
   .. automethod:: rank
   .. automethod:: rank_with_weights
   .. automethod:: register_sale
   .. automethod:: register_sale_failure
   .. automethod:: register_supply
   .. automethod:: register_supply_failure
   .. automethod:: reset_changes
   .. automethod:: sample_outcome_with_utility
   .. automethod:: scale_by
   .. automethod:: scale_max
   .. automethod:: scale_max_for
   .. automethod:: scale_min
   .. automethod:: scale_min_for
   .. automethod:: shift_by
   .. automethod:: shift_max_for
   .. automethod:: shift_min_for
   .. automethod:: spawn
   .. automethod:: spawn_object
   .. automethod:: to_crisp
   .. automethod:: to_dict
   .. automethod:: to_genius
   .. automethod:: to_prob
   .. automethod:: to_stationary
   .. automethod:: to_xml_str
   .. automethod:: utility_range
   .. automethod:: worst
   .. automethod:: xml
