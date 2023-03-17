===================
Stage XY - Template
===================

**Short statement of the stage's purpose**

`config template`_ 

.. _config template: configs/config_template.yaml

Input
=====
Describe type and format of the minimally required data and metadata.

*should pass* |check_input|_

.. |check_input| replace:: *check_input.py*
.. _check_input: scripts/check_input.py

Output
======
Describe type and format of the stage output data and metadata, eventual intermediate output, and where it is stored.

Usage
=====
Describe the functionality of the stage, what type of blocks are used and how they can be arranged, and eventually special stage parameters. However, an account of the exact blocks and their features should be placed into the Snakefile's and scripts' docstring.