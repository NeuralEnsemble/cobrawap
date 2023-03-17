{{ name | escape | underline}}
**{{ module | escape }}**

.. automodule:: {{ fullname }}

.. argparse::
   :module: {{ fullname }}
   :func: CLI
   :prog: {{ name }}
   :passparser: