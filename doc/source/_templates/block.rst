{{ fullname | escape | underline}}

.. argparse::
   :module: {{ fullname }}
   :func: create_parser
   :prog: {{ objname }}