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



