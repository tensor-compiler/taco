The `index_notation.h` file contains the API for tensor index notation, which are the way you define tensor computations in taco. Index expressions describe a tensor computation as a scalar expression where tensors are indexed by index variables. The index variabbles range over the tensor dimensions they index and the scalar expression is evaluated at each point.

Here are some examples:

```c++
// Matrix addition
A(i,j) = B(i,j) + C(i,j);

// Tensor addition (order-3 tensors)
A(i,j,k) = B(i,j,k) + C(i,j,k);

// Matrix-vector multiplication
a(i) = B(i,j) * c(j);

// Tensor-vector multiplication (order-3 tensor)
A(i,j) = B(i,j,k) * c(k);

// Matricized tensor times Khatri-Rao product (MTTKRP)
A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
```

In the `expr` API, tensors are instances of `TensorVar` and index variables of `IndexVar`. Here's a full example of how to use the expr API to define a sparse tensor-vector multiplication:

```c++
IndexVar i, j;

TensorVar A(Type(Double,{200,500}),     Format::CSR);
TensorVar B(Type(Double,{200,500,300}), Format::CSF);
TensorVar c(Type(Double,{300}),         Format::Dense);

A(i,j) = B(i,j,k) * c(k);
```
