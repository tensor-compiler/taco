#include "pyFormat.h"
#include "pybind11/stl.h"
#include <functional>

namespace taco{
namespace pythonBindings{

static inline std::size_t orInBit(std::size_t currentValue, int bitToSet){
  return currentValue | (1ULL << bitToSet);
}

// A hash function to satisfy python's requirement that objects that are equal should have the same hash value
static std::size_t hashModeFormat(const taco::ModeFormat& modeFormat){
  std::size_t hashValue = 0;
  hashValue = modeFormat.isFull()? orInBit(hashValue, 0): hashValue;
  hashValue = modeFormat.isOrdered()? orInBit(hashValue, 1): hashValue;
  hashValue = modeFormat.isUnique()? orInBit(hashValue, 2): hashValue;
  hashValue = modeFormat.isBranchless()? orInBit(hashValue, 3): hashValue;
  hashValue = modeFormat.isCompact()? orInBit(hashValue, 4): hashValue;
  hashValue = modeFormat.hasCoordPosIter()? orInBit(hashValue, 5): hashValue;
  hashValue = modeFormat.hasCoordValIter()? orInBit(hashValue, 6): hashValue;
  hashValue = modeFormat.hasLocate()? orInBit(hashValue, 7): hashValue;
  hashValue = modeFormat.hasInsert()? orInBit(hashValue, 8): hashValue;
  hashValue = modeFormat.hasAppend()? orInBit(hashValue, 9): hashValue;
  hashValue = modeFormat.defined()? orInBit(hashValue, 10): hashValue;

  std::hash<std::string> string_hash;
  return hashValue + string_hash(modeFormat.getName());
}

static std::size_t hashModeFormatPack(const taco::ModeFormatPack& modeFormatPack){
  const auto& modeTypes = modeFormatPack.getModeFormats();

  std::size_t hashValue = 0;
  for(int i = 0; i < static_cast<int>(modeTypes.size()); ++i){
    hashValue += (i+1)*hashModeFormat(modeTypes[i]);
  }

  return hashValue + 7*modeTypes.size();
}

std::size_t hashFormat(const taco::Format& format){

  const auto& modeTypePacks = format.getModeFormatPacks();
  const auto& ordering      = format.getModeOrdering();

  std::size_t hashValue = 0;
  for(int i = 0; i < static_cast<int>(ordering.size()); ++i){
    hashValue += hashModeFormatPack(modeTypePacks[i]) * (ordering[i] + 1);
  }

  return hashValue + 11 * ordering.size();
}

void defineModeFormats(py::module &m){

  py::class_<taco::ModeFormat>(m, "mode_format",  R"//(
Defines the storage format for a given dimension (mode) of a tensor.

Dimensions (modes) can either be dense (all elements are stored) or compressed as a sparse representation where
only the non-zeros of the dimension are stored.


Attributes
-----------
name

Examples
----------
>>> import pytaco as pt
>>> pt.dense
mode_format(dense)
>>> pt.compressed
mode_format(compressed)
>>> pt.dense.name
'dense'

Notes
----------
PyTaco currently exports the following mode formats:

:attr:`~pytaco.compressed` or :attr:`~pytaco.Compressed` - Only store non-zeros. eg. The second mode (dimension) in CSR

:attr:`~pytaco.dense` or :attr:`~pytaco.Dense` - Store all elements in dimension. eg. The first mode (dimension) in CSR


Explicit 0s resulting from computation are always stored even though a mode is marked as compressed. This is to avoid
checking every result from a computation which would slow down taco.
)//")
//          .def(py::init<>())
          .def_property_readonly("name", &taco::ModeFormat::getName,  R"//(
Returns a string identifying the mode format. This will either be 'compressed' or 'dense'
)//")
//          .def("is_full", &taco::ModeFormat::isFull)
//          .def("is_ordered", &taco::ModeFormat::isOrdered)
//          .def("is_unique", &taco::ModeFormat::isUnique)
//          .def("is_branchless", &taco::ModeFormat::isBranchless)
//          .def("is_compact", &taco::ModeFormat::isCompact)
//          .def("has_coord_val_iter", &taco::ModeFormat::hasCoordValIter)
//          .def("has_coord_pos_iter", &taco::ModeFormat::hasCoordPosIter)
//          .def("has_locate", &taco::ModeFormat::hasLocate)
//          .def("has_insert", &taco::ModeFormat::hasInsert)
//          .def("has_append", &taco::ModeFormat::hasAppend)
//          .def("defined", &taco::ModeFormat::defined)

          .def("__repr__", [](const taco::ModeFormat& modeFormat) -> std::string{
            std::ostringstream o;
            o << "mode_format(" << modeFormat << ")";
            return o.str();
          }, py::is_operator())

          .def("__eq__", [](const taco::ModeFormat& self, const taco::ModeFormat& other) -> bool{
              return self == other;
          }, py::is_operator())

          .def("__ne__", [](const taco::ModeFormat& self, const taco::ModeFormat& other) -> bool{
              return self != other;
          }, py::is_operator())

          .def("__hash__", [](const taco::ModeFormat &self) -> std::size_t {
              return hashModeFormat(self);
          }, py::is_operator());

  m.attr("Compressed") = taco::ModeFormat::Compressed;
  m.attr("compressed") = taco::ModeFormat::Compressed;
  m.attr("Dense") = taco::ModeFormat::Dense;
  m.attr("dense") = taco::ModeFormat::Dense;
}

void defineModeFormatPack(py::module& m){

  py::class_<taco::ModeFormatPack>(m, "mode_format_pack")
          .def(py::init<const std::vector<taco::ModeFormat>>())
          .def(py::init<const taco::ModeFormat>())
          .def("mode_formats", &taco::ModeFormatPack::getModeFormats)

          .def("__eq__", [](const taco::ModeFormatPack& self, const taco::ModeFormatPack other) -> bool{
            return self == other;
          }, py::is_operator())

          .def("__ne__", [](const taco::ModeFormatPack& self, const taco::ModeFormatPack& other) -> bool{
              return self != other;
          }, py::is_operator())

          .def("__hash__", [](const taco::ModeFormatPack &self) -> std::size_t {

              // Overflow doesn't affect python's required spec
              return hashModeFormatPack(self);
          }, py::is_operator())

          .def("__repr__", [](const taco::ModeFormatPack& self) -> std::string{
              std::ostringstream o;
              o << "mode_format_pack(" << self << ")";
              return o.str();
          }, py::is_operator());
}


void defineFormat(py::module &m){

  py::implicitly_convertible<taco::ModeFormat, taco::ModeFormatPack>();


  py::class_<taco::Format>(m, "format", R"//(
format(mode_formats=[], mode_ordering=[])

Create a :class:`~pytaco.tensor` format.

The modes have the given mode storage formats and are stored in the given sequence. Mode i has the :class:`mode_format`
specified by mode_formats[mode_ordering[i]].

If no arguments are given a format for a 0-order tensor (a scalar) is created.

Parameters
-----------

mode_formats: pytaco.mode_format, iterable of pytaco.mode_format, optional
    A list representing the mode format used to store each mode (dimension) of the tensor specified by mode_ordering[i].
    If a single :class:`~pytaco.mode_format` is given, then a format for a 1-order tensor (vector) is created. The
    default value is the empty list meaning a scalar is created.

mode_ordering: int, iterable of ints, optional
    Can be specified if len(mode_formats) > 1. Specifies the order in which the dimensions (modes) of the tensor
    should be stored in memory. That is, the mode stored in the i-th position in memory is specified by mode_ordering[i].
    Defaults to mode_ordering[i] = i which corresponds to row-major storage.


Notes
--------
PyTaco exports the following common formats:

:attr:`~pytaco.csr` or :attr:`~pytaco.CSR` - Compressed Sparse Row storage format.

:attr:`~pytaco.csc` or :attr:`~pytaco.CSC` - Compressed Sparse Columns storage format.

Attributes
-----------
order
mode_formats
mode_ordering

Examples
----------

Here, we will create two common storage formats CSR and CSC in order to better understand formats. First, we look at
CSR.

We need a mode formats list to tell taco the first dimension it stores should be dense and the second dimension
should be sparse.

>>> import pytaco as pt
>>> mode_formats = [pt.dense, pt.compressed]

We then need to tell taco the order in which to store the dimensions. Since we want CSR, we want to store the rows first
then the columns. Once we do this, we can make the format.

>>> mode_ordering = [0, 1] # Taco will default this if no ordering is given.
>>> csr = pt.format(mode_formats, mode_ordering)
>>> csr.order
2

Now, it is easy to make a CSC format given what we have already. For CSC, we want to store the columns before the rows
but also have the columns be dense and the rows be sparse. We do so as follows:

>>> mode_ordering_csc = [1,0]
>>> csc = pt.format(mode_formats, mode_ordering_csc)

This tells taco to store the columns before the rows due to the ordering given and to store the columns as dense since
they are now the first storage dimension and the mode_formats[0] is dense.

We can generalize this to make a large number of storage formats.
)//")
          .def(py::init<>())
          .def(py::init<const taco::ModeFormat>())
          .def(py::init<const std::vector<taco::ModeFormatPack> &>())
          .def(py::init<const std::vector<taco::ModeFormatPack> &, const std::vector<int> &>())
          .def_property_readonly("order",  &taco::Format::getOrder, R"//(
Returns the number of modes (dimensions) stored in a format.
)//")
          .def_property_readonly("mode_formats", &taco::Format::getModeFormats,R"//(
Returns the storage types of the modes. The type of mode stored in position i is specified by element i of the returned
vector.
)//")
//          .def("mode_format_packs", &taco::Format::getModeFormatPacks)
          .def_property_readonly("mode_ordering", &taco::Format::getModeOrdering, R"//(
Returns a list representing the ordering in which the modes are stored. The mode stored in position i is specified by
element i of the list returned.
)//")

//          .def("level_array_types", &taco::Format::getLevelArrayTypes)
//          .def("coordinate_type_pos", &taco::Format::getCoordinateTypePos)
//          .def("coordinate_type_idx", &taco::Format::getCoordinateTypeIdx)
//          .def("set_level_array_types", &taco::Format::setLevelArrayTypes)

          .def("__eq__", [](const taco::Format& self, const taco::Format other) -> bool{
              return self == other;
          }, py::is_operator())

          .def("__ne__", [](const taco::Format& self, const taco::Format& other) -> bool{
              return self != other;
          }, py::is_operator())

          .def("__hash__", [](const taco::Format &self) -> std::size_t {
              return hashFormat(self);
          }, py::is_operator())

          .def("__len__", &taco::Format::getOrder)

          .def("__repr__", [](const taco::Format& self) -> std::string{
              std::ostringstream o;
              o << "Format(" << self << ")";
              return o.str();
          }, py::is_operator());

  py::options options;
  options.disable_function_signatures();
  m.def("is_dense", &taco::isDense, R"//(
is_dense(fmt)

Checks if a format is all dense.

Parameters
-------------
    fmt: pytaco.format

Returns
---------
bool
    True of all dimensions (modes) in a tensor are stored in a dense format and False otherwise.

Examples
------------
>>> import pytaco as pt
>>> pt.is_dense(pt.csr)
False
>>> my_fmt = pt.format([pt.dense]*3)
>>> pt.is_dense(my_fmt)
True

)//");
  m.attr("CSR") = CSR;
  m.attr("csr") = CSR;
  m.attr("CSC") = CSC;
  m.attr("csc") = CSC;
}

}}