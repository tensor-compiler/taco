#include "hb2taco.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>

/*

  Reading and writing HB Harwell-Boeing Sparse File Format

*/

namespace hb2taco {

  void readFile(std::ifstream &hbfile, int** colptr, int** rowind, double** values){
    std::string title, key;
    int totcrd,ptrcrd,indcrd,valcrd,rhscrd;
    std::string mxtype;
    int nrow, ncol, nnzero, neltvl;
    std::string ptrfmt, indfmt, valfmt, rhsfmt;

    readHeader(hbfile,
	       &title, &key,
	       &totcrd, &ptrcrd, &indcrd, &valcrd, &rhscrd,
	       &mxtype, &nrow, &ncol, &nnzero, &neltvl,
	       &ptrfmt, &indfmt, &valfmt, &rhsfmt);

    if (*colptr)
      delete[] (*colptr);
    (*colptr) = new int[ncol+1];
    readIndices(hbfile, ptrcrd, *colptr);

    if (*rowind)
      delete[] (*rowind);
    (*rowind) = new int[nnzero];
    readIndices(hbfile, indcrd, *rowind);

    if (*values)
      delete[] (*values);
    (*values) = new double[nnzero];
    readValues(hbfile, valcrd, *values);

    readRHS();
  }

  void writeFile(std::ofstream &hbfile, std::string key,
		 int nrow, int ncol, int nnzero,
		 int ptrsize, int indsize, int valsize,
		 int* colptr, int* rowind, double* values){

    std::string title="CSC Matrix written by taco";
    int neltvl = 0;
    char mxtype[4] = "RUA";
    char indfmt[17] = "(16I5)";
    char ptrfmt[17] = "(16I5)";
    char rhsfmt[21] = "(10F7.1)";
    char valfmt[21] = "(10F7.1)";

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

  void readHeader(std::ifstream &hbfile,
		  std::string* title, std::string* key,
		  int* totcrd, int* ptrcrd, int* indcrd, int* valcrd, int* rhscrd,
		  std::string* mxtype, int* nrow, int* ncol, int* nnzero, int*neltvl,
		  std::string* ptrfmt, std::string* indfmt, std::string* valfmt, std::string* rhsfmt){
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
    std::getline(hbfile,line);
    /* Line 4 (2A16, 2A20)
    Col. 1 - 16 	Format for pointers (PTRFMT)
    Col. 17 - 32 	Format for row (or variable) indices (INDFMT)
    Col. 33 - 52 	Format for numerical values of coefficient matrix (VALFMT)
    Col. 53 - 72 	Format for numerical values of right-hand sides (RHSFMT) */
    iss.clear();
    iss.str(line);
    iss >> *ptrfmt >> *indfmt >> *valfmt >> *rhsfmt;
    if (*rhscrd > 0)
      std::getline(hbfile,line); // We wkip this line for taco
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
  void writeHeader(std::ofstream &hbfile,
		   std::string title, std::string key,
		   int totcrd, int ptrcrd, int indcrd, int valcrd, int rhscrd,
		   std::string mxtype, int nrow, int ncol, int nnzero, int neltvl,
		   std::string ptrfmt, std::string indfmt, std::string valfmt, std::string rhsfmt){

    hbfile << title  << " " << key << "\n";
    hbfile << totcrd << " " << ptrcrd << " " << indcrd << " " << valcrd << " " << rhscrd << "\n";
    hbfile << mxtype << " " << nrow   << " " << ncol   << " " << nnzero << " " << neltvl << "\n";
    hbfile << ptrfmt << " " << indfmt << " " << valfmt << " " << rhsfmt << "\n";
    // Last line useless for taco
  }

  void readIndices(std::ifstream &hbfile, int linesize, int indices[]){
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

  void writeIndices(std::ofstream &hbfile, int indsize, int indperline, int indices[]){
    for (auto i = 1; i <= indsize; i++) {
      hbfile << indices[i-1] + 1 << " ";
      if (i%indperline==0)
	hbfile << "\n";
    }
    if (indsize%indperline != 0)
      hbfile << "\n";
  }

  void readValues(std::ifstream &hbfile, int linesize, double values[]){
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

  void writeValues(std::ofstream &hbfile, int valuesize, int valperline, double values[]){
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

}
