#include "taco/storage/file_io_rb.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <climits>

#include "taco/tensor.h"
#include "taco/error.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/util/files.h"
#include "taco/util/collections.h"
#include "taco/cuda.h"

using namespace std;

namespace taco {

void readFile(std::istream &hbfile,
              int* nrow, int* ncol,
              int** colptr, int** rowind, double** values){
  std::string title, key;
  int totcrd,ptrcrd,indcrd,valcrd,rhscrd;
  std::string mxtype;
  int nnzero, neltvl;
  std::string ptrfmt, indfmt, valfmt, rhsfmt;

  readHeader(hbfile,
             &title, &key,
             &totcrd, &ptrcrd, &indcrd, &valcrd, &rhscrd,
             &mxtype, nrow, ncol, &nnzero, &neltvl,
             &ptrfmt, &indfmt, &valfmt, &rhsfmt);

  if (*colptr) {
    if (should_use_CUDA_unified_memory()) {
      cuda_unified_free(*colptr);
    }
    else {
      free(*colptr);
    }
  }
  if (should_use_CUDA_unified_memory()) {
    (*colptr) = (int*)cuda_unified_alloc((*ncol+1) * sizeof(int));
  }
  else {
    (*colptr) = (int*)malloc((*ncol+1) * sizeof(int));
  }
  readIndices(hbfile, ptrcrd, *colptr);

  if (*rowind) {
    if (should_use_CUDA_unified_memory()) {
      cuda_unified_free(*rowind);
    }
    else {
      free(*rowind);
    }
  }
  if (should_use_CUDA_unified_memory()) {
    (*rowind) = (int*)cuda_unified_alloc(nnzero * sizeof(int));
  }
  else {
    (*rowind) = (int*)malloc(nnzero * sizeof(int));
  }
  readIndices(hbfile, indcrd, *rowind);

  if (*values) {
    if (should_use_CUDA_unified_memory()) {
      cuda_unified_free(*values);
    }
    else {
      free(*values);
    }
  }
  (*values) = (double*)malloc(nnzero * sizeof(double));
  if (should_use_CUDA_unified_memory()) {
    (*values) = (double*)cuda_unified_alloc(nnzero * sizeof(double));
  }
  else {
    (*values) = (double*)malloc(nnzero * sizeof(double));
  }
  readValues(hbfile, valcrd, *values);

  readRHS();
}

void writeFile(std::ostream &hbfile, std::string key,
               int nrow, int ncol, int nnzero,
               int ptrsize, int indsize, int valsize,
               int* colptr, int* rowind, double* values){

  std::string title="CSC Matrix written by taco";
  int neltvl = 0;
  char mxtype[4] = "RUA";
  /*
    First Character:
    R Real matrix
    C Complex matrix
    P Pattern only (no numerical values supplied)

    Second Character:
    S Symmetric
    U Unsymmetric
    H Hermitian
    Z Skew symmetric
    R Rectangular

    Third Character:
    A Assembled
    E Elemental matrices (unassembled)
   */

  char indfmt[17] = "(16I5)";
  char ptrfmt[17] = "(16I5)";
  char valfmt[21] = "(10F7.1)";
  char rhsfmt[21] = "";

  int valcrd = valsize/10 + (valsize%10!=0);
  int ptrcrd = ptrsize/16 + (ptrsize%16!=0);
  int indcrd = indsize/16 + (indsize%16!=0);
  int rhscrd = 0;
  int totcrd = ptrcrd + indcrd + valcrd + rhscrd;

  writeHeader(hbfile,
              title, key,
              totcrd, ptrcrd, indcrd, valcrd, rhscrd,
              mxtype, nrow, ncol, nnzero, neltvl,
              ptrfmt, indfmt, valfmt, rhsfmt);

  writeIndices(hbfile, ptrsize, 16, colptr);

  writeIndices(hbfile, indsize, 16, rowind);

  writeValues(hbfile, valsize, 10, values);

  writeRHS();
}

void readHeader(std::istream &hbfile,
                std::string* title, std::string* key,
                int* totcrd, int* ptrcrd, int* indcrd, int* valcrd, int* rhscrd,
                std::string* mxtype, int* nrow,
                int* ncol, int* nnzero, int*neltvl,
                std::string* ptrfmt, std::string* indfmt,
                std::string* valfmt, std::string* rhsfmt){
  std::string line;
  std::getline(hbfile,line);
  /* Line 1 (A72,A8)
    Col. 1 - 72 	Title (TITLE)
    Col. 73 - 80 	Key (KEY) */
  std::istringstream iss(line);
  std::string word;
  while (iss >> word) {
    *title += " " + *key;
    if (iss >> *key) {
      *title += " " + word;
    }
  }
  std::getline(hbfile,line);
  /* Line 2 (5I14)
    Col. 1 - 14 	Total number of lines excluding header (TOTCRD)
    Col. 15 - 28 	Number of lines for pointers (PTRCRD)
    Col. 29 - 42 	Number of lines for row (or variable) indices (INDCRD)
    Col. 43 - 56 	Number of lines for numerical values (VALCRD)
    Col. 57 - 70 	Number of lines for right-hand sides (RHSCRD)
    	(including starting guesses and solution vectors if present)
    	(zero indicates no right-hand side data is present) */
  iss.clear();
  iss.str(line);
  iss >> *totcrd >> *ptrcrd >> *indcrd >> *valcrd >> *rhscrd;
  std::getline(hbfile,line);
  /* Line 3 (A3, 11X, 4I14)
    Col. 1 - 3 	Matrix type (see below) (MXTYPE)
    Col. 15 - 28 	Number of rows (or variables) (NROW)
    Col. 29 - 42 	Number of columns (or elements) (NCOL)
    Col. 43 - 56 	Number of row (or variable) indices (NNZERO)
    	(equal to number of entries for assembled matrices)
    Col. 57 - 70 	Number of elemental matrix entries (NELTVL)
    	(zero in the case of assembled matrices) */
  iss.clear();
  iss.str(line);
  iss >> *mxtype >> *nrow >> *ncol >> *nnzero >> *neltvl;
  taco_uassert((*mxtype).size() == 3 )
          << "mxtype in HBfile:  case not available " << *mxtype;
  /* First Character:
       R Real matrix
       C Complex matrix
       P Pattern only (no numerical values supplied) */
  taco_uassert(((*mxtype)[0] == 'R')||((*mxtype)[0] == 'r'))
          << "mxtype in HBfile:  case not available " << *mxtype;
  /* Second Character:
       S Symmetric
       U Unsymmetric
       H Hermitian
       Z Skew symmetric
       R Rectangular  */
  taco_uassert(((*mxtype)[1] == 'U')||((*mxtype)[1] == 'u'))
          << "mxtype in HBfile:  case not available " << *mxtype;
  /* Third Character:
       A Assembled
       E Elemental matrices (unassembled) */
  taco_uassert(((*mxtype)[2] == 'A')||((*mxtype)[2] == 'a'))
          << "mxtype in HBfile:  case not available " << *mxtype;
  std::getline(hbfile,line);
  /* Line 4 (2A16, 2A20)
    Col. 1 - 16     Format for pointers (PTRFMT)
    Col. 17 - 32    Format for row (or variable) indices (INDFMT)
    Col. 33 - 52    Format for numerical values of coefficient matrix (VALFMT)
    Col. 53 - 72    Format for numerical values of right-hand sides (RHSFMT) */
  iss.clear();
  iss.str(line);
  iss >> *ptrfmt >> *indfmt >> *valfmt;
  if (*rhscrd > 0) {
    iss >> *rhsfmt;
    std::getline(hbfile,line); // We wkip this line for taco
  }
  /* Line 5 (A3, 11X, 2I14) Only present if there are right-hand sides present
    Col. 1 	Right-hand side type:
    	F for full storage or
    	M for same format as matrix
    Col. 2 	G if a starting vector(s) (Guess) is supplied. (RHSTYP)
    Col. 3 	X if an exact solution vector(s) is supplied.
    Col. 15 - 28 	Number of right-hand sides (NRHS)
    Col. 29 - 42 	Number of row indices (NRHSIX)
    	(ignored in case of unassembled matrices) */
}
void writeHeader(std::ostream &hbfile,
                 std::string title, std::string key,
                 int totcrd, int ptrcrd, int indcrd, int valcrd, int rhscrd,
                 std::string mxtype, int nrow, int ncol, int nnzero, int neltvl,
                 std::string ptrfmt, std::string indfmt,
                 std::string valfmt, std::string rhsfmt){

  hbfile << title  << " " << key << "\n";
  hbfile << totcrd << " " << ptrcrd << " " << indcrd
         << " " << valcrd << " " << rhscrd << "\n";
  hbfile << mxtype << " " << nrow   << " " << ncol
         << " " << nnzero << " " << neltvl << "\n";
  hbfile << ptrfmt << " " << indfmt << " " << valfmt << " " << rhsfmt << "\n";
  // Last line useless for taco
}

void readIndices(std::istream &hbfile, int linesize, int indices[]){
  std::string line;
  std::string ptr;
  int ptr_ind=0;
  for (auto i = 0; i < linesize; i++) {
    std::getline(hbfile,line);
    std::istringstream iss(line);
    while(iss >> ptr) {
      indices[ptr_ind] = std::stoi(ptr) -1;
      ptr_ind++;
    }
  }
}

void writeIndices(std::ostream &hbfile, int indsize,
                  int indperline, int indices[]){
  for (auto i = 1; i <= indsize; i++) {
    hbfile << indices[i-1] + 1 << " ";
    if (i%indperline==0)
      hbfile << "\n";
  }
  if (indsize%indperline != 0)
    hbfile << "\n";
}

void readValues(std::istream &hbfile, int linesize, double values[]){
  std::string line;
  std::string ptr;
  int ptr_ind=0;
  for (auto i = 0; i < linesize; i++) {
    std::getline(hbfile,line);
    std::istringstream iss(line);
    while(iss >> ptr) {
      values[ptr_ind] = std::stod(ptr);
      ptr_ind++;
    }
  }
}

void writeValues(std::ostream &hbfile, int valuesize,
                 int valperline, double values[]){
  for (auto i = 1; i <= valuesize; i++) {
    if (std::floor(values[i-1]) == values[i-1])
      hbfile << values[i-1] << ".0 ";
    else
      hbfile << values[i-1] << " ";
    if (i%valperline==0)
      hbfile << "\n";
  }
  if (valuesize%valperline != 0)
    hbfile << "\n";
}

// Useless for Taco
void readRHS(){  }
void writeRHS(){  }

TensorBase readRB(std::string filename, const ModeFormat& modetype, bool pack) {
  taco_uassert(false) << "RB files must be loaded into a CSC matrix";
  return TensorBase();
}

TensorBase readRB(std::string filename, const Format& format, bool pack) {
  std::fstream file;
  util::openStream(file, filename, fstream::in);
  TensorBase tensor = readRB(file, format, pack);
  file.close();

  return tensor;
}

TensorBase readRB(std::istream& stream, const ModeFormat& modetype, bool pack) {
  taco_uassert(false) << "RB files must be loaded into a CSC matrix";
  return TensorBase();
}

TensorBase readRB(std::istream& stream, const Format& format, bool pack) {
  int rows, cols;
  int *colptr = NULL;
  int *rowidx = NULL;
  double *vals = NULL;

  readFile(stream, &rows, &cols, &colptr, &rowidx, &vals);

  taco_uassert(format == CSC) << "RB files must be loaded into a CSC matrix";
  TensorBase tensor(type<double>(), {(int)rows,(int)cols}, CSC);

  auto storage = tensor.getStorage();
  Index index(CSC,
              {ModeIndex({makeArray({(int)cols})}),
               ModeIndex({makeArray(colptr, cols+1, Array::Free),
                               makeArray(rowidx, colptr[cols], Array::Free)})});
  auto values = makeArray(vals, index.getSize(), Array::Free);

  storage.setIndex(index);
  storage.setValues(values);

  if (pack) {
    tensor.pack();
  }

  return tensor;
}

void writeRB(std::string filename, const TensorBase& tensor) {
  taco_iassert(tensor.getOrder() == 2) <<
      "The .rb format only supports matrices. Consider using the .tns format "
      "instead";

  std::fstream file;
  util::openStream(file, filename, fstream::out);
  writeRB(file, tensor);
  file.close();
}

void writeRB(std::ostream& stream, const TensorBase& tensor) {
  taco_uassert(tensor.getFormat() == CSC) <<
      "writeRB: the format of tensor " << tensor.getName() << " must be CSC";

  auto storage = tensor.getStorage();
  auto index = storage.getIndex();
  double *values = (double*)storage.getValues().getData();

  auto modeIndex = index.getModeIndex(1);
  auto colptr = modeIndex.getIndexArray(0);
  auto rowidx = modeIndex.getIndexArray(1);

  int nrow = tensor.getDimension(0);
  int ncol = tensor.getDimension(1);
  taco_iassert(index.getSize() <= INT_MAX);
  int nnzero = static_cast<int>(index.getSize());
  string key = tensor.getName();

  taco_iassert(colptr.getType() == type<int>());

  writeFile(stream,const_cast<char*> (key.c_str()),
            nrow,ncol,nnzero,
            static_cast<int>(colptr.getSize()),
            static_cast<int>(rowidx.getSize()), nnzero,
            (int*)colptr.getData(), (int*)rowidx.getData(), values);
}

}
