#ifndef TACO_ERROR_H
#define TACO_ERROR_H

#include <string>
#include <sstream>
#include <ostream>

namespace taco {

/// Error report (based on Halide's Error.h)
struct ErrorReport {
  enum Kind { User, Internal, Temporary };

  std::ostringstream *msg;
  const char *file;
  const char *func;
  int line;

  bool condition;
  const char *conditionString;

  Kind kind;
  bool warning;

  ErrorReport(const char *file, const char *func, int line, bool condition,
              const char *conditionString, Kind kind, bool warning);

  template<typename T>
  ErrorReport &operator<<(T x) {
    if (condition) {
      return *this;
    }
    (*msg) << x;
    return *this;
  }

  ErrorReport &operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (condition) {
      return *this;
    }
    (*msg) << manip;
    return *this;
  }

  ~ErrorReport() noexcept(false) {
    if (condition) {
      return;
    }
    explode();
  }

  void explode();
};

// internal asserts
#ifdef TACO_ASSERTS
  #define taco_iassert(c)                                                     \
    taco::ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,              \
                      taco::ErrorReport::Internal, false)
  #define taco_ierror                                                         \
    taco::ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,          \
                      taco::ErrorReport::Internal, false)
#else
  struct Dummy {
    template<typename T>
    Dummy &operator<<(T x) {
      return *this;
    }
    // Support for manipulators, such as std::endl
    Dummy &operator<<(std::ostream& (*manip)(std::ostream&)) {
      return *this;
    }
  };

  #define taco_iassert(c) taco::Dummy()
  #define taco_ierror taco::Dummy()
#endif

#define taco_unreachable                                                       \
  taco_ierror << "reached unreachable location"

// User asserts
#define taco_uassert(c)                                                        \
  taco::ErrorReport(__FILE__,__FUNCTION__,__LINE__, (c), #c,                   \
                    taco::ErrorReport::User, false)
#define taco_uerror                                                            \
  taco::ErrorReport(__FILE__,__FUNCTION__,__LINE__, false, nullptr,            \
                    taco::ErrorReport::User, false)
#define taco_uwarning                                                          \
  taco::ErrorReport(__FILE__,__FUNCTION__,__LINE__, false, nullptr,            \
                    taco::ErrorReport::User, true)

// Temporary assertions (planned for the future)
#define taco_tassert(c)                                                        \
  taco::ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,                 \
                    taco::ErrorReport::Temporary, false)
#define taco_terror                                                            \
  taco::ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,             \
                    taco::ErrorReport::Temporary, false)

#define taco_not_supported_yet taco_uerror << "Not supported yet"

}

#endif
