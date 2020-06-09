.. _iexpr:

Index Expressions
=======================

Index expressions are used to express computations in taco. They are formed by indexing into :class:`pytaco.tensor` s with
:class:`~pytaco.index_var` s and using the variety of arithmetic operations PyTaco provides for index vars.

It should be noted that index expressions describe to taco how a computation should be performed. Thus, they must be
assigned to a tensor before taco does any computations.

The documentation in this section displays all the functions taco supports with index expressions as well as a wide
number of examples for using them.

Index expressions provide a convenient way for taco to construct outputs without making temporaries and are essential
to getting leveraging taco's ability to fuse operations.

There is an equivalent tensor function for each index expression function. The user should note that the tensor functions
all use these under the hood.

.. toctree::
   :maxdepth: 3

   index_vars
   idx_exp_obj
   expr_funcs
   iv_funcs

