#include "taco/linalg.h"

using namespace std;

namespace taco {

// Just trying this out. Need to accept dimensions and format too.
/* LinalgBase::LinalgBase(Datatype ctype) */
/*   : LinalgBase(/1* get a unique name *1/, ctype) { */
/* } */

LinalgBase::LinalgBase(string name, Datatype ctype) : name(name), ctype(ctype), LinalgExpr(TensorVar(name, Type(ctype, {42,42}))) {

}




}
