#ifndef TACO_MODE_FORMAT_H
#define TACO_MODE_FORMAT_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>

#include "taco/ir/ir.h"
#include "taco/util/strings.h"

namespace taco {

class ModeFormat;
class ModeTypePack;

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

  struct ModePack;

  struct Mode {
    const ir::Expr      tensor;  // tensor containing mode
    const size_t    mode;    // identifier for mode
    const Dimension size;    // size of mode

    const ModePack* pack;    // reference to pack containing mode
    const size_t    pos;     // position within pack containing mode

    std::string getName() const;
  };

  struct ModePack {
    ModePack(const std::vector<Mode>& modes, 
             const std::vector<ModeType>& modeTypes);

    int getSize() const { return modes.size(); }

    ir::Expr getArray(size_t idx) const;

  private:
    std::vector<Mode> modes;
    std::vector<ModeType> modeTypes;
  };

  ModeType() = delete;
  ModeType(const std::shared_ptr<ModeFormat> modeFormat);
  ModeType(const ModeType& modeType);

  ModeType& operator=(const ModeType& modeType);
  ModeType operator()(const std::vector<Property>& properties = {});

  std::string getFormatName() const;

  bool isFull() const; 
  bool isOrdered() const; 
  bool isUnique() const; 
  bool isBranchless() const; 
  bool isCompact() const; 

  bool hasCoordValueIter() const; 
  bool hasCoordPosIter() const; 
  bool hasLocate() const;
  bool hasInsert() const;
  bool hasAppend() const;

private:
  std::shared_ptr<const ModeFormat> impl;

  // list of functions that need access to impl
  //friend std::vector<ir::Stmt> lower();
};

class ModeFormat {
public:
  ModeFormat() = delete;
  ModeFormat(const std::string formatName, const bool isFull, 
             const bool isOrdered, const bool isUnique, const bool isBranchless, 
             const bool isCompact, const bool hasCoordValueIter, 
             const bool hasCoordPosIter, const bool hasLocate, 
             const bool hasInsert, const bool hasAppend);

  virtual ~ModeFormat() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const = 0;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, 
      const ModeType::Mode& mode) const;
  
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev, const ModeType::Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, 
      const ModeType::Mode& mode) const;

  virtual ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual ir::Expr getSize(const ir::Expr& szPrev, 
      const ModeType::Mode& mode) const;
  virtual ir::Stmt getInsertInit(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;
  virtual ir::Stmt getInsertFinalize(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;
  
  virtual ir::Stmt getAppendCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd, const ModeType::Mode& mode) const;
  virtual ir::Stmt getAppendInit(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;
  virtual ir::Stmt getAppendFinalize(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;

  virtual ir::Expr getArray(size_t idx, const ModeType::Mode& mode) const = 0;

  const std::string formatName;

  const bool isFull;
  const bool isOrdered;
  const bool isUnique;
  const bool isBranchless;
  const bool isCompact;

  const bool hasCoordValueIter;
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

