Command Line Scripts
====================

When installing SCML through the pip command, you get one command line tool that can be used to
aid your development and testing. This tool provides a unified interface to all scml commands.

The set of supported commands are:

===============       ===================================================================
 Command                                  Meaning
===============       ===================================================================
tournament2019        Runs a tournament with SCML2019 settings
tournament2020        Runs a tournament with SCML2020 settings
run2019               Runs a 2019 tournament
run2020               Runs a 2020 tournament
version               Prints SCML version (and NegMAS version)
===============       ===================================================================


Arguments

========= ============================================
Argument   Meaning
========= ============================================
--help     Display a help screen
--gui      Runs the script as a GUI
========= ============================================

Running SCML2020 Tournaments
----------------------------

Runs a tournament using SCML2020 settings.  You can get help on this tool by running:

.. code-block:: console

    $ scml tournament2020 --help

These are the *optional* arguments of this tool:

========================================== ==============================================================
  Argument                                      Meaning
========================================== ==============================================================
  -n, --name TEXT                           The name of the tournament. The special
                                            value "random" will result in a random name
                                            [default: random]
  -s, --steps INTEGER                       Number of steps. If passed then --steps-min
                                            and --steps-max are ignored  [default: 10]
  --ttype, --tournament-type, --tournament  [collusion|std]
                                            The config to use. It can be collusion or
                                            std  [default: std]
  -t, --timeout INTEGER                     Timeout the whole tournament after the
                                            given number of seconds (0 for infinite)
                                            [default: -1]
  --configs INTEGER                         Number of unique configurations to
                                            generate.  [default: 5]
  --runs INTEGER                            Number of runs for each configuration
                                            [default: 2]
  --max-runs INTEGER                        Maximum total number of runs. Zero or
                                            negative numbers mean no limit  [default:
                                            -1]
  --competitors TEXT                        A semicolon (;) separated list of agent
                                            types to use for the competition. You can
                                            also pass the special value default for the
                                            default builtin agents  [default: Decentral
                                            izingAgent;BuyCheapSellExpensiveAgent;Rando
                                            mAgent]
  --non-competitors TEXT                    A semicolon (;) separated list of agent
                                            types to exist in the worlds as non-
                                            competitors (their scores will not be
                                            calculated).  [default: ]
  -l, --log DIRECTORY                       Default location to save logs (A folder
                                            will be created under it)  [default:
                                            ~/logs/tournaments]
  --world-config FILE                       A file to load extra configuration
                                            parameters for world simulations from.
  --verbosity INTEGER                       verbosity level (from 0 == silent to 1 ==
                                            world progress)  [default: 1]
  --log-ufuns / --no-ufun-logs              Log ufuns into their own CSV file. Only
                                            effective if --debug is given  [default:
                                            False]
  --log-negs / --no-neg-logs                Log all negotiations. Only effective if
                                            --debug is given  [default: False]
  --compact / --debug                       If True, effort is exerted to reduce the
                                            memory footprint whichincludes reducing
                                            logs dramatically.  [default: True]
  --raise-exceptions / --ignore-exceptions  Whether to ignore agent exceptions
                                            [default: True]
  --path TEXT                               A path to be added to PYTHONPATH in which
                                            all competitors are stored. You can path a
                                            : separated list of paths on linux/mac and
                                            a ; separated list in windows  [default: ]
  --cw INTEGER                              Number of competitors to run at every world
                                            simulation. It must either be left at
                                            default or be a number > 1 and < the number
                                            of competitors passed using --competitors
                                            [default: 3]
  --parallel / --serial                     Run a parallel/serial tournament on a
                                            single machine  [default: True]
  --config FILE                             Read configuration from FILE.
  --help                                    Show help message.
========================================== ==============================================================

Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the `--log` argument (default is negmas/logs/tournaments under the HOME directory). To avoid over-writing
earlier results, a new folder will be created for each run named by the current date and time. The
folder will contain the following files:


=========================   ========     =================================================================
 File/Folder Name             Format         Content
=========================   ========     =================================================================
configs                     FOLDER       Contains one json file for each world
                                         run tried during the tournament. You can
                                         re-run this world using `run_world` function in the `tournament`
                                         module.
params.json                 JSON         The parameters used to create this tournament
base_configs.json           JSON         The base configurations used in the tournament (without agent/factory
                                         assignments.
assigned_configs.json       JSON         The configurations used after assigning factories to managers
scores.csv                  CSV          Scores of every agent in every world
total_scores.csv            CSV          Scores of every agent **type** averaged over all runs
winners.csv                 CSV          Winner *types* and their average scores
ttest.csv                   CSV          Results of a factorial TTEST comparing the performance of all
                                         agent *types*
=========================   ========     =================================================================

Other than these files, a folder with the same number as the corresponding config file in the configs folder, keeps full
statistics/log of every world *but only if --debug is specified* (see the `SCML2020World Runner` section for the contents of
this folder.

Running SCML2019 Tournaments
----------------------------

Runs a tournament using SCML2019 settings.  You can get help on this tool by running:

.. code-block:: console

    $ scml tournament2019 --help

These are the *optional* arguments of this tool:

========================================== ==============================================================
  Argument                                      Meaning
========================================== ==============================================================
  -n, --name TEXT                           The name of the tournament. The special
                                            value "random" will result in a random name
                                            [default: random]
  -s, --steps INTEGER                       Number of steps. If passed then --steps-min
                                            and --steps-max are ignored  [default: 10]
  --ttype, --tournament-type, --tournament  [collusion|std]
                                            The config to use. It can be collusion or
                                            std  [default: std]
  -t, --timeout INTEGER                     Timeout the whole tournament after the
                                            given number of seconds (0 for infinite)
                                            [default: -1]
  --configs INTEGER                         Number of unique configurations to
                                            generate.  [default: 5]
  --runs INTEGER                            Number of runs for each configuration
                                            [default: 2]
  --max-runs INTEGER                        Maximum total number of runs. Zero or
                                            negative numbers mean no limit  [default:
                                            -1]
  --competitors TEXT                        A semicolon (;) separated list of agent
                                            types to use for the competition. You can
                                            also pass the special value default for the
                                            default builtin agents  [default: Decentral
                                            izingAgent;BuyCheapSellExpensiveAgent;Rando
                                            mAgent]
  --non-competitors TEXT                    A semicolon (;) separated list of agent
                                            types to exist in the worlds as non-
                                            competitors (their scores will not be
                                            calculated).  [default: ]
  -l, --log DIRECTORY                       Default location to save logs (A folder
                                            will be created under it)  [default:
                                            ~/logs/tournaments]
  --world-config FILE                       A file to load extra configuration
                                            parameters for world simulations from.
  --verbosity INTEGER                       verbosity level (from 0 == silent to 1 ==
                                            world progress)  [default: 1]
  --log-ufuns / --no-ufun-logs              Log ufuns into their own CSV file. Only
                                            effective if --debug is given  [default:
                                            False]
  --log-negs / --no-neg-logs                Log all negotiations. Only effective if
                                            --debug is given  [default: False]
  --compact / --debug                       If True, effort is exerted to reduce the
                                            memory footprint whichincludes reducing
                                            logs dramatically.  [default: True]
  --raise-exceptions / --ignore-exceptions  Whether to ignore agent exceptions
                                            [default: True]
  --path TEXT                               A path to be added to PYTHONPATH in which
                                            all competitors are stored. You can path a
                                            : separated list of paths on linux/mac and
                                            a ; separated list in windows  [default: ]
  --cw INTEGER                              Number of competitors to run at every world
                                            simulation. It must either be left at
                                            default or be a number > 1 and < the number
                                            of competitors passed using --competitors
                                            [default: 3]
  --parallel / --serial                     Run a parallel/serial tournament on a
                                            single machine  [default: True]
  --config FILE                             Read configuration from FILE.
  --help                                    Show help message.
========================================== ==============================================================


Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the `--log` argument (default is negmas/logs/tournaments under the HOME directory). To avoid over-writing
earlier results, a new folder will be created for each run named by the current date and time. The
folder will contain the following files:


=========================   ========     =================================================================
 File/Folder Name             Format         Content
=========================   ========     =================================================================
configs                     FOLDER       Contains one json file for each world
                                         run tried during the tournament. You can
                                         re-run this world using `run_world` function in the `tournament`
                                         module.
params.json                 JSON         The parameters used to create this tournament
base_configs.json           JSON         The base configurations used in the tournament (without agent/factory
                                         assignments.
assigned_configs.json       JSON         The configurations used after assigning factories to managers
scores.csv                  CSV          Scores of every agent in every world
total_scores.csv            CSV          Scores of every agent **type** averaged over all runs
winners.csv                 CSV          Winner *types* and their average scores
ttest.csv                   CSV          Results of a factorial TTEST comparing the performance of all
                                         agent *types*
=========================   ========     =================================================================

Other than these files, a folder with the same number as the corresponding config file in the configs folder, keeps full
statistics/log of every world *but only if --debug is specified* (see the `SCML2020World Runner` section for the contents of
this folder.

Running an SCML2020 world (scml run2020)
----------------------------------------

Runs a single world simulation of SCML2020.

================================================ =======================================================
  Parameter                                         Meaning
================================================ =======================================================
  --steps INTEGER                                 Number of steps.  [default: 10]
  --time INTEGER                                  Total time limit.  [default: 7200]
  --competitors TEXT                              A semicolon (;) separated list of agent
                                                  types to use for the competition.  [default: RandomAgent]
  --log DIRECTORY                                 Default location to save logs (A folder will
                                                  be created under it)  [default: ~/negmas/logs]
  --log-ufuns / --no-ufun-logs                    Log ufuns into their own CSV file. Only
                                                  effective if --debug is given  [default: False]
  --log-negs / --no-neg-logs                      Log all negotiations. Only effective if
                                                  --debug is given  [default: False]
  --compact / --debug                             If True, effort is exerted to reduce the
                                                  memory footprint whichincludes reducing logs
                                                  dramatically.  [default: False]
  --raise-exceptions / --ignore-exceptions        Whether to ignore agent exceptions [default: True]
  --path TEXT                                     A path to be added to PYTHONPATH in which
                                                  all competitors are stored. You can path a :
                                                  separated list of paths on linux/mac and a ;
                                                  separated list in windows  [default: ]
  --world-config FILE                             A file to load extra configuration
                                                  parameters for world simulations from.
  --config FILE                                   Read configuration from FILE.
  --help                                          Show help and exit.
================================================ =======================================================


Upon completion of the simulation, logs and statistics are stored in the log folder specified by `--log` argument or under '~/negmas/logs/scml/scml2020/{date-time}'. The following files and folders can be found there:

=========================   ========     =================================================================
 File/Folder Name             Format         Content
=========================   ========     =================================================================
 agents.json                 JSON           Basic information about all the agents in the simulation.
 all_contracts.csv           CSV            Details of all contracts saved by the system
 info.json                   JSON           Details of the parameters used for world geenration
 params.json                 JSON           The result of running vars() on the world. It contains mostly
                                            the paramters used to genrate the world plus some of its
                                            final stats
 log.txt                    TXT             Contains logs
 negotiations.csv           CSV             Details of all negotiations run during the simulation
 stats.csv                  CSV             Stats kept by the world (same as what you get from running
                                            save_stats)
 stats.json                 JSON            Same data as in stats.csv (for backword compatibility)
=========================   ========     =================================================================

Running an SCML2019 world (scml run2019)
----------------------------------------

Runs a single world simulation of SCML2019.

================================================ =======================================================
  Parameter                                         Meaning
================================================ =======================================================
  --steps INTEGER                                 Number of steps.  [default: 10]
  --time INTEGER                                  Total time limit.  [default: 7200]
  --competitors TEXT                              A semicolon (;) separated list of agent
                                                  types to use for the competition.
                                                  [default: GreedyFactoryManager]
  --jcompetitors, --java-competitors TEXT         A semicolon (;) separated list of agent
                                                  types to use for the competition.
                                                  [default: ]
  --log DIRECTORY                                 Default location to save logs (A folder will
                                                  be created under it)  [default: ~/negmas/logs]
  --log-ufuns / --no-ufun-logs                    Log ufuns into their own CSV file. Only
                                                  effective if --debug is given  [default: False]
  --log-negs / --no-neg-logs                      Log all negotiations. Only effective if
                                                  --debug is given  [default: False]
  --compact / --debug                             If True, effort is exerted to reduce the
                                                  memory footprint whichincludes reducing logs
                                                  dramatically.  [default: False]
  --raise-exceptions / --ignore-exceptions        Whether to ignore agent exceptions [default: True]
  --path TEXT                                     A path to be added to PYTHONPATH in which
                                                  all competitors are stored. You can path a :
                                                  separated list of paths on linux/mac and a ;
                                                  separated list in windows  [default: ]
  --world-config FILE                             A file to load extra configuration
                                                  parameters for world simulations from.
  --config FILE                                   Read configuration from FILE.
  --help                                          Show this message and exit.
================================================ =======================================================


Upon completion of the simulation, logs and statistics are stored in the log folder specified by `--log` argument or under '~/negmas/logs/scml/scml2020/{date-time}'. The following files and folders can be found there:

=========================   ========     =================================================================
 File/Folder Name             Format         Content
=========================   ========     =================================================================
 agents.json                 JSON           Basic information about all the agents in the simulation.
 all_contracts.csv           CSV            Details of all contracts saved by the system
 info.json                   JSON           Details of the parameters used for world geenration
 params.json                 JSON           The result of running vars() on the world. It contains mostly
                                            the paramters used to genrate the world plus some of its
                                            final stats
 log.txt                    TXT             Contains logs
 negotiations.csv           CSV             Details of all negotiations run during the simulation
 stats.csv                  CSV             Stats kept by the world (same as what you get from running
                                            save_stats)
 stats.json                 JSON            Same data as in stats.csv (for backword compatibility)
=========================   ========     =================================================================
