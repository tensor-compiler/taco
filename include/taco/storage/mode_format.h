#ifndef TACO_MODE_FORMAT_H
#define TACO_MODE_FORMAT_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>
#include <map>

#include "taco/ir/ir.h"
#include "taco/util/strings.h"

namespace taco {
namespace lower {
class Iterators;
}

namespace storage {
class IteratorImpl;
}

class ModeFormat;
class ModeTypePack;

struct ModePack;

class ModeType {
public:
  static ModeType dense;
  static ModeType compressed;
  static ModeType sparse;

  static ModeType Dense;
  static ModeType Compressed;
  static ModeType Sparse;

  enum Property {
    FULL, NOT_FULL, ORDERED, NOT_ORDERED, UNIQUE, NOT_UNIQUE, BRANCHLESS, 
    NOT_BRANCHLESS, COMPACT, NOT_COMPACT
  };

  ModeType();
  ModeType(const std::shared_ptr<ModeFormat> modeFormat);
  ModeType(const ModeType& modeType);

  ModeType& operator=(const ModeType& modeType);
  ModeType operator()(const std::vector<Property>& properties = {});
  
  bool defined() const;

  std::string getFormatName() const;

  bool isFull() const; 
  bool isOrdered() const; 
  bool isUnique() const; 
  bool isBranchless() const; 
  bool isCompact() const; 

  bool hasCoordValIter() const; 
  bool hasCoordPosIter() const; 
  bool hasLocate() const;
  bool hasInsert() const;
  bool hasAppend() const;

private:
  std::shared_ptr<const ModeFormat> impl;

  // list of functions that need access to impl
  //friend std::vector<ir::Stmt> lower();
  friend struct ModePack;
  friend class storage::IteratorImpl;
};

struct Mode {
  Mode(const ir::Expr tensor, const size_t mode, const Dimension size, 
       const ModePack* const pack, const size_t pos, 
       const ModeType prevModeType);

  const ir::Expr        tensor;        // tensor containing mode
  const size_t          mode;          // identifier for mode
  const Dimension       size;          // size of mode

  const ModePack* const pack;          // reference to pack containing mode
  const size_t          pos;           // position within pack containing mode
  const ModeType        prevModeType;  // format of previous mode

  std::string getName() const;

  ir::Expr getVar(const std::string varName) const;
  bool     hasVar(const std::string varName) const;
  void     addVar(const std::string varName, ir::Expr var);

private:
  std::map<std::string, ir::Expr> vars;
};

struct ModePack {
  size_t getSize() const;

  ir::Expr getArray(size_t idx) const;

private:
  std::vector<Mode> modes;
  std::vector<ModeType> modeTypes;

  friend class lower::Iterators;
};

class ModeFormat {
public:
  ModeFormat() = delete;
  ModeFormat(const std::string formatName, const bool isFull, 
             const bool isOrdered, const bool isUnique, const bool isBranchless, 
             const bool isCompact, const bool hasCoordValIter, 
             const bool hasCoordPosIter, const bool hasLocate, 
             const bool hasInsert, const bool hasAppend);

  virtual ~ModeFormat() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const = 0;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i, Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, Mode& mode) const;
  
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev, Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, Mode& mode) const;
  
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, Mode& mode) const;

  virtual ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, Mode& mode) const;
  virtual ir::Expr getSize(Mode& mode) const;
  virtual ir::Stmt getInsertInitCoords(const ir::Expr& pBegin, 
      const ir::Expr& pEnd, Mode& mode) const;
  virtual ir::Stmt getInsertInitLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  virtual ir::Stmt getInsertFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  
  virtual ir::Stmt getAppendCoord(const ir::Expr& p, const ir::Expr& i, 
      Mode& mode) const; 
  virtual ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd, Mode& mode) const;
  virtual ir::Stmt getAppendInitEdges(const ir::Expr& pPrevBegin, 
      const ir::Expr& pPrevEnd, Mode& mode) const;
  virtual ir::Stmt getAppendInitLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  virtual ir::Stmt getAppendFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;

  virtual ir::Expr getArray(size_t idx, const Mode& mode) const = 0;

  const std::string formatName;

  const bool isFull;
  const bool isOrdered;
  const bool isUnique;
  const bool isBranchless;
  const bool isCompact;

  const bool hasCoordValIter;
  const bool hasCoordPosIter;
  const bool hasLocate;
  const bool hasInsert;
  const bool hasAppend;
};

bool operator==(const ModeType&, const ModeType&);
bool operator!=(const ModeType&, const ModeType&);

std::ostream& operator<<(std::ostream&, const ModeType&);

}
#endif

