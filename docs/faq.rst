===
FAQ
===

How can I access a data file in my package?
-------------------------------------------

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


