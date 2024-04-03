Changelog
=========

0.7.4 (2024.04.03)
------------------

* AWI: Correcting n_competitors and adding my_competitors
* slightly faster random oneshot
* bugfix in builtin_std_agents
* Adding more contexts
* Adding loading sklearn to FAQ

0.7.3 (2024.03.15)
------------------

* Requiring negmas 0.10.18

0.7.2 (2024.03.02)
------------------

* requiring negmas >=0.10.17
* minor bugfix correcting method name in 2022 version
* correcting NMI type in some agents
* bugfix CLI failed because "method" is not in World
* WorldRunner improvements: Allowing separation of agent addition and running
* WorldRunner improvements: Controlling order of plotting in runner.
* replacing lambdas in AWI to allow pickling
* adding names to contexts
* avoiding testing exact decoding in obs manager

0.7.1 (2024.03.02)
------------------

* Adding WorldRunner to simplify experimenting with new agent designs
* Compatibility with negmas 0.10.16
* Adding more contexts to oneshot and standard
* improved documentation

0.7.0 (2024.03.02)
------------------

* Major improvements to RL support
* Added more contexts that make more sense.
* Improved support of SCML OneShot environments in FlexibleActionManager
* Added FlexibleObservationManager which supports any SCML OneShot
* environment.
* SCML Std is not well supported by RL components right now
* Moving context out of rl
* Adding plot_stats to world and plot_combined_stats in world
* supporting py3.12 again
* switching to ruff  for formatting
* Improving implementations of greedy and random for Std
* removing MIP from requirements
* adding dynamic switching to RLAgent
* Making contexts callable (and minor other changes)
* Adding access to future supply and demand in AWI.  In OneShotAWI if you are not using perishable products, you can now access future supply and demand (according to your signed contracts) using self.awi.future_*
* Small fixes for OneShotEnv registration and warnings raised by UnconstrainedActionManager
* bugfix: incorrect setting of perishable in std
* bugfixes in adapter and inheriting std

0.6.3 (2023.12.11)
------------------

* typing update
* removing unnecessary sphinxtesters
* requiring latest stable_baselines
* doc update
* bugfix in scml2020 adapter to run oneshot
* Adding skeleton for the new std simulation
* Adding 2024 utility functions and worlds
* Switching docs to scml.readthedocs.io

0.6.2 (2023.11.27)
------------------

* upgrading CI to python 3.12
* importing observations
* dropping support for python < 3.10
* Completing first draft of the RL environment

0.6.1 (2023.7.9)
----------------

* Moving scml.scml2020.utils to scml.utils (to avoid a circular import in rl/factory)
* This is an initial effort to have RL work for scml. Currently we are disabling the extra checks on observation and
  world construction because they lead to failures. This needs to be rectified later.
* Added the concept of a WorldFactory to generate worlds with predefined characteristics from the point of view of the learning agent.
* Now environments, action managers, and observation managers all receive WorldFactory objects and automatically check for compatibility.

0.6.0 (2023.7.9)
----------------

* Upgrading to be compatible with NegMAS 0.10.0
* Adding step_with() to oneshot allowing for single counter-offer set stepping of the simulation (to be used to expose the simulation as RL and MARL environments later).
* Adding current_negotiation_details to OneShotAWI to get details of running negotiation
* Adding managed sales, supplies, total_sales, total_supplies, needed_sales, and needed_supplies to the OneShotAWI
* Extending OneShotState returned from awi.state and awi.default_state_encoder(mechanism_states) to expose more information about the simulation

0.5.6 (2023.3.8)
----------------

* Upgrading to be compatible with NegMAS 0.9.8
* Updating notebooks for 2023 release
* Adding 2023 specific classes and utility functions
* Supporting most recent negmas version on Github
* Adding the ability to use raw collusion scores
	In the collusion track, it is now possible to control how much does the
	raw collusion score affect the final score relative to the difference
	between raw collusion score and standard score.
* more reobust renaming when reveal-names is set

0.5.5 (2022.5.18)
-----------------
* bugfix in anac2022_* methods: Score reporting was not accurate because of a change in naming convention in negmas.
* adding agent-processes to SCML2022World.generate
* Adding SCML2022World which is just an alias for SCML2021World
* bugfix #96 Partial List of Agent Scores: Also discussion $94
    * Incorrect agent types and mysterious agents appearing in the scores
      list.
    * Inaccurate placement of agents in the simulation
* disabling tests taking too long on CI
* commenting members of Profile, Contract in oneshot

0.5.4 (2022.5.15)
-----------------
* Disabling adapter testing. Adapting SCML2020 and OneShot agents is too much effort and is not reallly that useful. We will eventually remove support for that.
* Improving control over oneshot world generation. SCML2020OneShotWorld.generat() method now is more consistent in the way it creates agents.
* Adding wide_price_range to control the unit-price range

0.5.3 (2022.5.14)
-----------------
* randomizing starting agents in oneshot
* improving AWI typing in oneshot
* Resetting agents before making negotiations
*   - before_step() was also resetting agents in oneshot. This meant that all
*   negotiators that were created in _make_negotiations() were deleted in
*   reset().
*   - Now we call  reset() before making negotaitions and before_step() after
*   that.
*   - This guarantees that the agent has access to its negotiators in
*   before_step() as in 2020.
* Adding new collusion evaluation
* Goal: Disentangle the quality of the collusion strategy and the standard
* strategy.
* Method: Each agent now has 3 factories in collusion. We run four
* simulations:
* - s0: The agent controls all of the three factories
* - s1-s3: The agent controls one of them
* The final score of the agent is its score in s0 minus the average score
* it got in s1-s3
* Avoid calling counter_all before first_offers
* github actions update
* improved testing of sync behavior
* Confirming that OneShotSync gets all offers every counter_all() call and upper limiting the difference in negotiation speeds based on Jackson's code

0.5.2 (2022.4.8)
----------------

* Minor fixes in the utilities module to simplify running anac 2022 tournaments.

0.5.1 (2022.3.10)
-----------------

* giving more time to testing 2020 with random
* avoiding a hashing issue in second tutorial
* rejecting crazy offers in satisficer
* ensuring that from_offers recevied tuple[tuple]
* oneshto ufun speedup
* adding ANAC 2022 running functions scml2020.utils
* import cleanup
* control sync_calls when creating oneshot worlds
* fix: requirements.txt to reduce vulnerabilities

0.5.0 (2022.2.17)
-----------------

* Compatibility with NegMAS 0.9.0

0.4.9 (2021.7.30)
-----------------

* [bugfix] #73 ufun was sometimes one-step back

0.4.8 (2021.7.28)
-----------------

* [bugfix] Bankrupt agents kepts requesting negotiations
* [bugfix] trading prices inaccurate during step
* [std] Adding spot_market_quantity/loss to the AWI
* [oneshot] Adding helpers to avoid crazy prices

0.4.7 (2021.7.15)
-----------------

* [std/collusion] keeping compatibility with 2020
* [std/collusion] Avoiding crash when agent class names is shorter than 2
* letters
* [std/collusion] rare division by zero error
* [oneshot] forcing n_processes to 2 in tournaments

0.4.6 (2021.6.15)
-----------------

*  [all tracks] Adding convenience methods to AWIs. closes #49
*  [all tracks] reducing production cost range
*  [all tracks] casting offers to ints explicitly to avoid bugs in agents offering fractional quantities/unit_prices
*  [oneshot] avoiding ultimatum in oneshot (was disabled)
*  [oneshot] Add `current-inventory` to OneshotAWI which will always return zero for compatibility with the other tracks
*  [onesht] adding public_* to summary.
*  [oneshot] penalty scale was incaccurate
*  [std/collusion] Adding satisficer agent
*  [std/collusion] adding guarnteed_profit method of world generation
*  [std/collusion] better handling of predictions in builtin
*  [std/collusion] experimental better decentralizing agent
*  [std/collusion] Increasing profit potential
*  [speedup] avoid saving negotiations online

0.4.5 (2021.6.14)
-----------------

* [oneshot] refactoring using before_step in greedy
* [minor] formating update
* [bugfix] production costs were not increasing.  closes #38
* [API] Adding before_step() to all agents. If a method called `before_step()` is defined for an agent, it will be called once every simulated day before any other calls to the agent but after all exogenous contracts and ufun parameters are fixed.  Note that `step()` is called at the *end* not the beginning of the day.
* [oneshot] limiting exogenous quantities to n lines. This closes #37
* [oneshot] Aspiration negotiator gets more agreements
* [oneshot] allowing agents to skip their turn. Agents can skip their turn now by returning REJECT_OFFER, None
* [oneshot] GreedyOneShotAgent is more rational now
* [bugfix] avoiding an inconsistency in path names for logs
* [oneshot] improved builtin agents (aspiration)
* [tournament] avoiding a possible edge case that would have led to competitor agents appearing as non-competitors
* [tournament] Saving negotiatinos by default in the CLI and utils
* [std/coll] defaulting to narrower worlds

0.4.4 (2021.6.1)
----------------

* [oneshot] Matching default parameters of world generation to game description.

0.4.3 (2021.6.1)
----------------

* [oneshot] Adding OneshotIndNegotiatorsAgent to use independent negotiators in oneshot
* [std/coll] updating builtin compoenents to be more rational
* [bugfix] failure in distributing products when the number of agents per process becomes large.
* [cli] changes on default competitors in the CLI
* [oneshot] adding the option to disable avoid-ultimatum (and disabling it for now)
* [oneshot] making sure needs are integers in greedy
* [bugfix] negotiator id is not the same as partner id sometimes
* [cli] adding --name to run2021 command to control world name
* [bugfix] negotiator_id and partner_id were not equal
* [eval] ensuring that ageents are run in exactly the same conditions
* [eval] adding zscore, iqr, fraction (old iqr -> iqr_fraction) to truncated  mean
* [cli] changing default n. competitors to 2
* [oneshot] adding an option not to count/not count future contracts on bankrruptcy

0.4.2 (2021.5.10)
-----------------
* [bugfix] Avoiding an error if an agent gave a fractional unit-price
* [bugfix] avoiding a test failure in CI that cannt be reproduced (I hate doing that :-( )
* [bugfix] added current_inventory to 2020's awi fixes $31
* [cli] Changing default agents for oneshot in cli
* [tournament] adding truncated-mean as an evaluation criterion and making it the default for scml2021

0.4.1 (2021.5.2)
-----------------
**This is an important update. All participants in SCML 2021 should upgrade
to this version**

*  [visualizer] adding run information for the visualizer
*  [bugfix] Std agents running in OneShot were able to request selling from the wrong agents.
*  [docs] doc update (storage cost -> disposal, deilvery penalty -> shortfall)
*  [core] supporting 3.9
*  [oneshot] better optimized ufun calculation
*  [oneshot] improved ufun calculation. Still not exact.
*  [2021] Adding current_balance to all AWIs and using it in oneshot ufun
*  [2020] Exporting AWI, Failure from scml2020.world for backward comp.

0.4.0 (2021.3.18)
-----------------
**This is an important update. All participants in SCML 2021 should upgrade
to this version**

* compatibility with negmas 0.8.0
* [oneshot] bugfix in random negotiator with ami is None
* [scml2020] all market aware agents work now and are parametrized
* [onshot] calculating ufun limits only for normalized ufuns.  Agents now MUST
* call find_limit() explicitly on the ufun to calculate limits except for ufuns
* created passing normalized=True in which find_limit() is called to calculate
* best and worst in construction.
* [oneshot][bugfix] my_consumers was wrong issue fix #13
* [docs] documentation update
* [oneshot] adding running_negotiations and unsigned_contracts
* [oneshot] changing breach conditions
* [docs] Update README.rst

0.3.4 (2021.3.8)
-----------------

* compatibility with negmas 0.7.4
* minor bugfixes

0.3.3 (2021.2.22)
-----------------
**This is an important update. All participants in SCML 2021 should upgrade to this version**

* Allowing std/collusion agents to run in OneShot track
* Adding min_utility, max_utility to ufun
* Adding exogenous_*_predictability parameters
* Using these parameters SCML2020/2021/OneShot worlds can be configured so that the exogenous contracts of the same agents at different time-steps are predictable (or not). If predictability is zero then each agent can have any quantity for its exogenous contracts. If predictability is 1.0 then it will have the same quantity at every time-step and if it is somewhere in between, the quantity at different steps will be similar to each other.
* Allowing OneShot agents to run in std track
* balance in one-shot plus bug fixes (lots of them)
* Adding oneshot module for SCML2020-Oneshot track
* Adding disallow_concurrent_negs_with_same_partners If given, avoids concurrent negotiations between the same partners.  Avoiding adding assignment number twice to world names
* Adding an upper/lower limit on buying/selling prices
* Adding extra scores for collusion league
* Do not sign clearly bad contracts (Decnetralizing)
* Dropping contracts with time >= current step.  These were already being dropped but may not have had a dropped_at value
* Never save videos/logs in tournaments
* avoiding test failure if PyQT was not installed
* Explicitly dropping invalid contracts
* Contracts with 0 price/quantity are nullified

0.3.0 (2020.7.2)
----------------
**This is an important update. All participants in SCML 2020 should upgrade to this version**

* [bugfix] Production cost is not properly discounted. This is an important issue.
  All simulations were conducted using the same zero production cost for all factories.
* Speeding up tournament tests (smaller worlds)
* consistent naming of non-competitors
* Adding dynamic choice of non-competitors
* Removing random from the set of default agents
* Compatibility with NegMAS 0.6.14

0.2.14 (2020.5.05)
------------------

* [docs] documentation and testing update.
* [setup] Making PyQT optional.
* [setup] Requiring negmas 0.6.13.

0.2.13 (2020.4.15)
------------------

* [docs] Adding more tutorials
* minor. Maing the controller optional in request_negotiations
* adding score to FactorySimulator to estimate final score

0.2.12 (2020.4.13)
------------------

* forcing negmas 0.6.11 or newer
* documentation update
* enabling setting the mechanism parameters in SCML2020World
* bugfix in PredictionBasedTradingStrategy

0.2.11 (2020.3.29)
------------------

*  bugfix in the CLI when running tournament2019
*  bugfix in MeanERPrediction for breached contracts
*  making CheapBuyer compatible with the latest negmas version
*  doc update
*  removing all agent logs in built-in agents to speedup simulations
*  changing cli script name back to cli.py. This was done to avoid a weird import error when running configs that use the cli in pycharm
*  adding profiling info snapshot to the repository

0.2.10 (2020.3.25)
------------------

* minor updates to be compatible with the latest negmas
* documentation update
* avoid exception if gui is not installed

0.2.9 (2020.3.19)
-----------------

* CI using Github Actions
* consolidating tests outside src directory
* Adding advanced script and upgrading negmas
* removing unnecessary init function from simulator
* showing shorter names in tournament run results

0.2.8 (2020.3.13)
-----------------
* documentation update (specially the scripts section)
* Adding a --gui option to scml CLI to run it as a simple GUI
* Simplifying the parameters of SCML CLI by keeping only the onese that
  do not conflict with the default parameters used in the competition
* adding a script call scmladv.py which keep all the detailed parameters
  used earlied in SCML.

0.2.7 (2020.3.09)
-----------------
* Documentation update
* Adding trading_strategy_init/step functions.
* Correcting a bug in n_competitors_per_world.
* allowing control of the number of participants per simulation explicitly in scml2020
* [SCML2020] Activating negotiation step quota
* [Doc] Adding a tutorial about logs and stats
* correcting the display in scml run2020
* removing the docs from the package to save space

0.2.6 (2020.2.27)
-----------------

* [testing] correcting a test to ignore system agents when checking for
  bankruptcy
* [minor] Reformating using Black
* [bugfix] Resolving a but in the CLI tournament command that prevented it from
  running with default parameters

0.2.5 (2020.2.27)
-----------------

* [Documentation] Removing inherited members to make the documentation easier to
  follow
* [CLI] Improving the display of run2020 command

0.2.4 (2020.2.21)
-----------------

* [speed] improvement in tournament running
* [bugfix] handling very short simulations

0.2.3 (2020.2.15)
-----------------

* adding more details to tournament runs
* update to tournament utilities of SCML2020
* doc update and correcting a bug in world generation
* better initialization of production graph depth
* correcting default factory assignments (if any)
* making do_nothing agent really do nothing in scml2020
* removing unnecessary assertion
* correcting world generation using the new width first approach
* correcting world generation using the new width first approach
* documentation update
* adding no_logs option to SCMLWorld2019 and SCMLWorld2020
* changing default logging location for SCML2019 and SCML2020 to ~/negmas/logs/tournament
* changing the way worlds are generated in SCML2020 to minimize the number of agents per level allowing the depth to increase
* removing built docs from the repository
* modification to .gitignore
* updating .gitignore


0.2.2 (2020.1.31)
-----------------

* adding components
* adding second tutorial

0.2.1 (2020.1.23)
-----------------

* better tutorial
* better strategies

0.2.0 (2020.1.8)
----------------

* new interface for singing and callbacks
* new interface for exogenous contracts
* improved decentralizing strategy

0.1.5 (2019.12.11)
------------------

* correcting tournament implementation for SCML2020
* updates to SCML2019 agents to be compatible with newer versions of negmas

0.1.3 (2019-12-08)
------------------

* adding run2020 to scml commands (see the command line tool's documentation)
* Now tournaments run for SCML 2020 configuration

0.1.2 (2019-12-01)
------------------

* Adding SCML 2020 simulation.


0.1.1 (2019-11-25)
------------------

* Adding all agents from SCML 2019 competition to the `scml2019` package.
* Adding first draft of SCML 2020 implementation.

0.1.0 (2019-11-20)
------------------

* First release on PyPI.
