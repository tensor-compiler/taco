#ifndef TAC_ERROR_H
#define TAC_ERROR_H

#include <string>
#include <sstream>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cassert>


#include <exception>
#include <string>
#include <vector>
#include <iostream>

namespace tac {
namespace internal {

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

  // Support for manipulators, such as std::endl
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
#ifdef TAC_ASSERTS
  #define iassert(c)                                                         \
    tac::internal::ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,    \
                               tac::internal::ErrorReport::Internal, false)
  #define ierror                                                             \
    tac::internal::ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,\
                               tac::internal::ErrorReport::Internal, false)
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

  #define iassert(c) tac::internal::Dummy()
  #define ierror tac::internal::Dummy()
#endif

#define unreachable                                                            \
  ierror << "reached unreachable location"

// internal assert helpers
#define iassert_scalar(a)                                                      \
  iassert(isScalar(a.type())) << a << ": " << a.type()

#define iassert_types_equal(a,b)                                               \
  iassert(a.type() == b.type()) << a.type() << " != " << b.type() << "\n"      \
                                << #a << ": " << a << "\n" << #b << ": " << b

#define iassert_int_scalar(a)                                                  \
  iassert(isScalar(a.type()) && isInt(a.type()))                               \
      << a << "must be an int scalar but is a" << a.type()

#define iassert_boolean_scalar(a)                                              \
  iassert(isScalar(a.type()) && isBoolean(a.type()))                           \
      << a << "must be a boolean scalar but is a" << a.type()

// User asserts
#define uassert(c)                                                             \
  tac::internal::ErrorReport(__FILE__,__FUNCTION__,__LINE__, (c), #c,        \
                             tac::internal::ErrorReport::User, false)
#define uerror                                                                 \
  tac::internal::ErrorReport(__FILE__,__FUNCTION__,__LINE__, false, nullptr, \
                             tac::internal::ErrorReport::User, false)
#define uwarning                                                               \
  tac::internal::ErrorReport(__FILE__,__FUNCTION__,__LINE__, false, nullptr, \
                             tac::internal::ErrorReport::User, true)

// Temporary assertions (planned for the future)
#define tassert(c)                                                             \
  tac::internal::ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,      \
                             tac::internal::ErrorReport::Temporary, false)
#define terror                                                                 \
  tac::internal::ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,  \
                             tac::internal::ErrorReport::Temporary, false)

#define not_supported_yet terror

}}

#endif
