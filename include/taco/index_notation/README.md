The file `index_notation.h` contains the API of the index notation
language for describing computations on tensors.  We support three
index notation dialects that share the same data structures: einsum
notation, reduction notation, and concrete notation.  Einsum notation
is a subset of reduction notation, and reduction notation is a subset
of concrete notation.

Index notation describes a tensor computation as a scalar assignment,
where tensors are indexed by index variables that range over a domain
of index values.  The scalar expression is evaluated at each point in
this domain and assigned to corresponding location in the result.

In the API, `TensorVar` represent tensors and `IndexVar` index
variables. Here's a full example of how to use the einsum dialect of
the API to define a sparse tensor-vector multiplication:

```c++
#include <iostream>
#include "taco.h"
using namespace taco;

int main(int argc, char* argv[]) {
  IndexVar i, j;

  TensorVar A(Type(Double,{M,N}),   CSR);
  TensorVar B(Type(Double,{M,N,O}), CSF);
  TensorVar c(Type(Double,{O}),     Dense);

  A(i,j) = B(i,j,k) * c(k);
  std::cout << A << std::endl;
}
```

# Einsum Notation

The einsum index notation dialect describes tensor computations as an
assignment from a scalar expression to a tensor.  Index variables that
are not used to index the result are called reduction or summation
variables and are summed over the term they are used in.

Here are some einsum index notation examples:
```c++
// Matrix addition
A(i,j) = B(i,j) + C(i,j);

// Tensor addition
A(i,j,k) = B(i,j,k) + C(i,j,k);

// Matrix-vector multiplication
a(i) = B(i,j) * c(j);

// Tensor-vector multiplication
A(i,j) = B(i,j,k) * c(k);

// Matricized tensor times Khatri-Rao product (MTTKRP)
A(i,j) = B(i,k,l) * C(l,j) * D(k,j);
```

Index notation that has reduction variables without explicit
reductions is considered an einsum notation and cannot have any
reductions.


# Reduction Notation

The reduction index notation dialect, which we often just refer to as
index notation, adds explicit reduction nodes.  In reduction notation
every reduction variable must have explicit reduction nodes.

Here are some reduction index notation examples:
```c++
// Matrix addition
A(i,j) = B(i,j) + C(i,j);

// Tensor addition
A(i,j,k) = B(i,j,k) + C(i,j,k);

// Matrix-vector multiplication
a(i) = sum(j, B(i,j) * c(j));

// Tensor-vector multiplication
A(i,j) = sum(k, B(i,j,k) * c(k));

// Matricized tensor times Khatri-Rao product (MTTKRP)
A(i,j) = sum(k, sum(l, B(i,k,l) * C(l,j) * D(k,j)));
```


# Concrete Notation

The concrete index notation dialect adds additional index statement
beyond the single assignment, and removes the reduction nodes in favor
of compound/incrementing assignments.  These statements describe when
the different scalar sub-expressions are computed and where they are
stored (result or temporary variables).  The purpose of concrete
notation is to express computations and it is described in more detail
in the [optimization paper](https://arxiv.org/abs/1802.10574).

**Most users will not need to use concrete notation, but can instead
use einsum or reduction notation together with scheduling operations
(see below).**

The index notation statements supported by concrete notation are:

- An **assignment** statement assigns an index expression to the
  locations in a tensor given by an lhs access expression.
- A **forall** statement binds an index variable to values and evaluates
  the sub-statement for each of these values.
- A **where** statment has a producer statement that binds a tensor
  variable in the environment of a consumer statement.

Here are some concrete index notation examples:
```c++
// Matrix addition (row-major)
forall(i,
       forall(j,
              A(i,j) = B(i,j) + C(i,j) ));

// Tensor addition
forall(i,
       forall(k,
              forall(j,
                     A(i,j,k) = B(i,j,k) + C(i,j,k) )));

// Matrix-vector multiplication
forall(i,
       forall(j,
              a(i) += B(i,j) * c(j) ));

// Tensor-vector multiplication (with dense workspace to scatter values into)
forall(i,
       forall(j,
              where(forall(k,
                           A(i,j) = w(k)),
                    forall(k,
                           w(k) += B(i,j,k) * c(k) ))));

// Matricized tensor times Khatri-Rao product (MTTKRP) (with workspace)
forall(i,
       forall(k,
              where(forall(j,
                           A(i,j) += w(j) * D(k,j)),
                    forall(l,
                           forall(j,
                                  w(j) += B(i,k,l) * C(l,j) )))));
```


# Scheduling Language
TBD
