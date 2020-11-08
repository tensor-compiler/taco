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
    // Unpack the type and shape
    Datatype type = tensorType.getDataType();
    Shape shape = tensorType.getShape();
    vector<Dimension> dimensions(shape.begin(), shape.end());
    vector<int> dims;
    for(const Dimension& d : dimensions) {
      dims.push_back((int)d.getSize());
    }

    // Init a TensorBase
    tbase = new TensorBase(name, type, dims, format);

    cout << "Created TensorBase " << tbase->getName() << endl;
    cout << tbase << endl;
}


LinalgAssignment LinalgBase::operator=(const LinalgExpr& expr) {
  taco_iassert(isa<VarNode>(this->ptr));
  cout << "LinalgBase operator= on " << name << endl;
  LinalgAssignment assignment = LinalgAssignment(dynamic_cast<const VarNode*>(this->ptr)->tensorVar, expr);
  /* cout << "this assignment ptr: " << this->assignment.ptr << endl; */
  this->assignment = assignment;
  /* cout << "this assignment ptr: " << this->assignment.ptr << endl; */
  
  // Now that the assignment is made we should run the index-assignment algorithm
  
  // Start by trying to print out the whole expression tree

  return assignment;
}

}
