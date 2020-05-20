===
FAQ
===

How can I access a data file in my package?
-------------------------------------------

When your agent is submitted, it is run in an environment different from that in which the tournament
will be run. This means that you **cannot** use hardcoded paths in your agent. Moreover, you (and we) do
not know in advance what will be the current directory when the tournament is run. For this reason, it is
**required** that if you access any files in your agent, you should use a path relative to the file in which
the code accessing these files is located. Please note that accessing ANY FILES outside the directory of
your agent is **prohibited** and **will lead to immediate disqualification** for obvious security reasons.
There are no second chances in this one.

Let's assume that your file structure is something like that:

::

    base
    ├── sub
    │   ├── myagent.py
    │   └── otherfiles.py
    ├── data
    │   └── myfile.csv
    └── tests


Now you want to access the file *myfile.csv* when you are inside *myagent.py*. To do so you can use the following code::

    import pathlib
    path_2_myfile = pathlib.Path(__file__).parent.parent / "data" / "myfile.csv"

Can my agent pass data to my other agents between worlds?
---------------------------------------------------------
**NO** Passing data to your agents between world simulations will lead to
disqualification.

Can my agent read data from the HDD outside my agent's folder?
--------------------------------------------------------------
**NO** Your agent can only read files that you submitted to us in your zip file.
It cannot modify these files in anyway during the competition.
It cannot read from anywhere else in secondary storage. Trying to do
so will lead to disqualification.

Can my agent write data to the HDD during the simulation?
---------------------------------------------------------
**NO** The agent is not allowed to write anything to the hard disk during the
competition.

Can I print to the screen to debug my agent?
--------------------------------------------
**PLEASE DO NOT**

Printing to the screen in your agent will prevent us from monitoring the progress of tournament
runs and will slow down the process. Moreover, it is not useful anyway because the tournaments are run in
parallel.

If you really need to print something (e.g. for debugging purposes), you can use one of the following two
methods:

    1. Remove all print statements before submission. We will never touch your code after submission so we cannot remove
       them.
    2. Use the screen logging facility provided by negmas. When creating a world (or a tournament) pass the following
       parameter::

          import logging
          World(..., log_screen_level=logging.DEBUG)
          # or create_tournament/anac2020std/anac2020collusion/tournament(..., log_screen_level=logging.DEBUG)

          # You can then use something like athis to log to the screen (and file)
          self.awi.logdebug("MY LOG MESSAGE")


Can I write arbitrary code in my module besides the agent class definition?
---------------------------------------------------------------------------
When python imports your module, it runs everything in it so the top level code should be only one of these:

    - Class definitions
    - Function definitions
    - Variable definitions
    - Arbitrary code that runs in few milliseconds and prints nothing

Any other code *must* be protected inside::

    if __name__ == "__main__"

For example, if you want to run a simulation to test your agent. *DO NOT USE SOMETHING LIKE THIS*::

    w = SCML2020World(....)
    w.run()

But something like this::

    def main():
        w = SCML2020World(...)
        w.run()

    if __name__ == "__main__":
        main()

This way, importing your module will not run the world simulation.

I ran a simulation using "scml run2020" command. Where are my log files?
------------------------------------------------------------------------

If you did not pass a log location through "--log", you will find the log files
at *~/negmas/logs/scml/scml2020/[date-time-uuid]*


I implement my agent using multiple files. How should I import them?
--------------------------------------------------------------------

Assume that you have the following file structure

::

    base
    ├── subfolder
    │   └── component2.py
    ├── component1.py
    └── agent.py

In your `agent.py` file, you want to import your other files::

    import component1
    import subfolder.component2

This will **not** work because in the actual competition `component1.py` and
`component2.py` will not be in python path.

There are two ways to solve it:

The clean way is to use relative imports. You will need to turn your agent int a package
by adding empty `__init__.py` files to every folder you want to import from::

    base
    ├── sub
    │   ├── __init__.py
    │   └── component2.py
    ├── __init__.py
    ├── component1.py
    └── agent.py

You can now change your import to::

    import .component1
    import .subfolder.component2

Notice the extra dot (`.`)

Another way that does not require any modification of your file structure is to add the following lines
**before** your imports::

    import os, sys
    sys.path.append(os.path.dirname(__file__))

Note that the later method has the disadvantage of putting your components at the **end** of the path which
means that if you have any classes, functions, etc with a name that is defined in *any* module that appears
earlier in the path, yours will be hidden.

How can I run simulations with the same parameters as the actual competition (e.g. for training my agent)
---------------------------------------------------------------------------------------------------------

You can use the `utils` submodule of `scml.scml2020` to generate worlds with the same parameters as in the
competition. Here is some example script to run `1` such world using three built-in agents::

    from typing import List, Union
    from scml.scml2020.utils import (
        anac2020_config_generator,
        anac2020_world_generator,
        anac2020_assigner,
    )
    from scml.scml2020 import SCML2020Agent
    from scml.scml2020.agents import (
        DecentralizingAgent, BuyCheapSellExpensiveAgent, RandomAgent
    )

    COMPETITORS = [DecentralizingAgent, BuyCheapSellExpensiveAgent, RandomAgent]

    def generate_worlds(
        competitors: List[Union[str, SCML2020Agent]],
        n_agents_per_competitor,
    ):
        collusion = n_agents_per_competitor != 1
        config = anac2020_config_generator(
            n_competitors=len(competitors),
            n_agents_per_competitor=n_agents_per_competitor,
            n_steps=(50, 100) if collusion else (50, 200),
        )
        assigned = anac2020_assigner(
            config,
            max_n_worlds=None,
            n_agents_per_competitor=n_agents_per_competitor,
            competitors=competitors,
            params=[dict() for _ in competitors],
        )
        return [anac2020_world_generator(**(a[0])) for a in assigned]

    if __name__ == "__main__":
        worlds = generate_worlds(COMPETITORS, 1)
        for world in worlds:
            world.run()
            print(world.stats_df.head())

Notice that `generate_worlds` will not generate a single world but a set of them putting the `COMPETITORS`
in all possible assignments of factories. The detailed process of world generation is described in the appendices of the
`description
<http://www.yasserm.com/scml/scml2020.pdf>`_ .

You can change the competitors by just changing the `COMPETITORS` list. Setting the third parameter of `generate_worlds`
to `1` generates a standard league world and setting it to a random number between 2 and 4 generates a collusion
league world ( `randint(2, min(4, len(COMPETITORS)))` ).
