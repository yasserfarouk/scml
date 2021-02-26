Changelog
=========

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
