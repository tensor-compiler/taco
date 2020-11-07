#include "taco/linalg.h"

using namespace std;

namespace taco {

// Just trying this out. Need to accept dimensions and format too.
/* LinalgBase::LinalgBase(Datatype ctype) */
/*   : LinalgBase(/1* get a unique name *1/, ctype) { */
/* } */

LinalgBase::LinalgBase(string name, Type tensorType) : LinalgExpr(TensorVar(name, tensorType)), name(name), tensorType(tensorType)
  {
}
LinalgBase::LinalgBase(string name, Type tensorType, Format format) : LinalgExpr(TensorVar(name, tensorType, format)), name(name), tensorType(tensorType)
  {
}

LinalgAssignment LinalgBase::operator=(const LinalgExpr& expr) {
  taco_iassert(isa<VarNode>(this->ptr));
  cout << "LinalgBase operator=" << endl;
  LinalgAssignment assignment = LinalgAssignment(dynamic_cast<const VarNode*>(this->ptr)->tensorVar, expr);
  this->assignment = assignment;
  return assignment;
}

}
