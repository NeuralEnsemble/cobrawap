{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. argparse::
   :module: {{ fullname }}
   :func: create_parser
   :prog: {{ name }}
