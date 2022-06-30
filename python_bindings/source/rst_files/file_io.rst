.. _io:

Tensor I/O
=================================

PyTaco can read and write tensors natively from three different file formats.
The formats PyTaco supports are:

- `Matrix Market (Coordinate) Format <https://math.nist.gov/MatrixMarket/formats.html>`_ - .mtx
- `Rutherford-Boeing Format <https://www.cise.ufl.edu/research/sparse/matrices/DOC/rb.pdf>`_ - .rb
- `FROSTT Format <http://frostt.io/tensors/file-formats.html>`_ - .tns

For both the read and write functions, the file format is inferred from the
file extension. This means that the extension must be given when specifying a
file name.

.. currentmodule:: pytaco

.. autosummary::
   :toctree: functions

   read
   write
   from_array
   from_sp_csc
   from_sp_csr
   to_array
   to_sp_csc
   to_sp_csr
   as_tensor
