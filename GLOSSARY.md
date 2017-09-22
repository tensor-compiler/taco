A library for compiling tensor algebra expressions combines ideas from several
fields, so there is bound to be some confusion over what certain words mean.
For this reason, we list here the meaning of several important words throughout
the codebase.

# Glossary
### Component
  A **component** is a single object which we wish to store.

In other contexts, components have been referred to as "entries," or "elements."

### Tensor Basics
  A **tensor** is a mathematical object which associates each list of *K*
integers *(i<sub>0</sub>, i<sub>1</sub>, ..., i<sub>K - 1</sub>)* with a
component, where *0 ≤ i<sub>k</sub> \< N<sub>k</sub>* for all *0 ≤ k < K*.

### Coordinate
  A **coordinate** is the list of *K* integers *(i<sub>0</sub>, i<sub>1</sub>,
..., i<sub>K - 1</sub>)* in the definition of a tensor which is used to
describe a particular component.

In other contexts, coordinates have been referred to as "indices."
### Sparse Tensor
  A tensor whose components are almost all zeros is said to be **sparse**. We
can save memory when storing a sparse tensor by storing only nonzero components
and their corresponding components.

### Mode
  A **mode** is a position in the list of integers which make up a coordinate.

In other contexts, the mode has been referred to as a "dimension," or an "axis."

### Index
  An **index** is an integer in the list of integers which make up a coordinate.

  An **index** is also a word we use to describe a datastructure which stores
nonzero coordinates in a tensor, or which stores nonzero indices in a mode.

### Dimension
  The **dimension** of a mode is the number of indices in that mode. In the
definition of the tensor, we have that the dimension of mode *k* is
*N<sub>k</sub>* since *0 ≤ i<sub>k</sub> \< N<sub>k</sub>*>

### Order
  The **order** of a tensor is the number of modes in the tensor. This is also
the number of indices necessary to create a coordinate which describes a tensor
component. The order of the tensor in the tensor definition is *K*, and would
be described as a "tensor of order *K*".

  In other contexts, the order has been referred to as the "rank."

### Ordering
  When storing a sparse tensor in software, it can sometimes be useful to
permute the modes before storing coordinates. The **ordering** of the modes
within a tensor is the sequence in which the modes are stored in the
tensor data structure.

### Format
  There are many ways to represent the coordinates of a tensor which correspond
to nonzero components. We refer to a particular representation scheme for an
index as its **format**.
