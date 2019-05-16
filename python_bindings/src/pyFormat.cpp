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

  py::class_<taco::ModeFormat>(m, "modeFormat")
          .def(py::init<>())
          .def("name", &taco::ModeFormat::getName)
          .def("is_full", &taco::ModeFormat::isFull)
          .def("is_ordered", &taco::ModeFormat::isOrdered)
          .def("is_unique", &taco::ModeFormat::isUnique)
          .def("is_branchless", &taco::ModeFormat::isBranchless)
          .def("is_compact", &taco::ModeFormat::isCompact)
          .def("has_coord_val_iter", &taco::ModeFormat::hasCoordValIter)
          .def("has_coord_pos_iter", &taco::ModeFormat::hasCoordPosIter)
          .def("has_locate", &taco::ModeFormat::hasLocate)
          .def("has_insert", &taco::ModeFormat::hasInsert)
          .def("has_append", &taco::ModeFormat::hasAppend)
          .def("defined", &taco::ModeFormat::defined)

          .def("__repr__", [](const taco::ModeFormat& modeFormat) -> std::string{
            std::ostringstream o;
            o << "ModeFormat(" << modeFormat << ")";
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

  py::class_<taco::ModeFormatPack>(m, "modeFormatPack")
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
              o << "ModeFormatPack(" << self << ")";
              return o.str();
          }, py::is_operator());
}


void defineFormat(py::module &m){

  py::implicitly_convertible<taco::ModeFormat, taco::ModeFormatPack>();

  py::class_<taco::Format>(m, "format")
          .def(py::init<>())
          .def(py::init<const taco::ModeFormat>())
          .def(py::init<const std::vector<taco::ModeFormatPack> &>())
          .def(py::init<const std::vector<taco::ModeFormatPack> &, const std::vector<int> &>())
          .def("order",  &taco::Format::getOrder)
          .def("mode_formats", &taco::Format::getModeFormats)
          .def("mode_format_packs", &taco::Format::getModeFormatPacks)
          .def("mode_ordering", &taco::Format::getModeOrdering)
          .def("level_array_types", &taco::Format::getLevelArrayTypes)
          .def("coordinate_type_pos", &taco::Format::getCoordinateTypePos)
          .def("coordinate_type_idx", &taco::Format::getCoordinateTypeIdx)
          .def("set_level_array_types", &taco::Format::setLevelArrayTypes)

          .def("__eq__", [](const taco::Format& self, const taco::Format other) -> bool{
              return self == other;
          }, py::is_operator())

          .def("__ne__", [](const taco::Format& self, const taco::Format& other) -> bool{
              return self != other;
          }, py::is_operator())

          .def("__hash__", [](const taco::Format &self) -> std::size_t {
              return hashFormat(self);
          }, py::is_operator())

          .def("__repr__", [](const taco::Format& self) -> std::string{
              std::ostringstream o;
              o << "Format(" << self << ")";
              return o.str();
          }, py::is_operator());

  m.def("is_dense", &taco::isDense);
}

}}