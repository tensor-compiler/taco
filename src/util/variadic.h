#ifndef TACO_UTIL_VARIADIC_H
#define TACO_UTIL_VARIADIC_H

#include <type_traits>

namespace taco {
namespace util {

// Compare types
template<class T, class...>
struct areSame : std::true_type {};

template<class T1, class T2, class... TN>
struct areSame<T1, T2, TN...>
  : std::integral_constant<bool, std::is_same<T1,T2>{} && areSame<T1, TN...>{}>
{};


// Product
namespace {

template <int...>
struct productHelper;

template <int prod, int val, int... rest>
struct productHelper<prod, val, rest...> {
  static const int value = productHelper<prod * val, rest...>::value;
};


template <int val>
struct productHelper<val> {
  static const int value = val;
};

} // unnamed namespace

template <int... vals> struct product;

template <>
struct product <> {
  static constexpr int value = 1;
};

template <int val, int... rest>
struct product <val, rest...> {
  static const int value = productHelper<1, val, rest...>::value;
};

template <int... vals>
struct product {
  static const int value = product<vals...>::value;
};


// Machinery to compute array offsets
template <int...> struct seq {};

/// Remove first value from int variadic template
template <int first, int... rest>
struct removeFirst {
  typedef seq<rest...> type;
};


/// Compute product of static sequence
template <int... values> inline constexpr
int computeProduct(seq<values...> seq) {
  return product<values...>::value;
}


/// Compute the offset into an n-dimensional array
template <int... dimensions, typename... Indices> inline
int computeOffset(seq<dimensions...> dims, int index, Indices... rest) {
  typename removeFirst<dimensions...>::type innerDims;
  return index * computeProduct(innerDims) + computeOffset(innerDims, rest...);
}

template <int... dimensions> inline constexpr
int computeOffset(const seq<dimensions...> &dims, int i) {
  return i;
}


/// Compute the offset into an n-dimensional array
template <int... dimensions> inline
int computeOffset(seq<dimensions...> dims, const std::vector<size_t>& indices) {
  return computeOffset(dims, indices.begin(), indices.end());
}

template <int... dimensions> inline
int computeOffset(seq<dimensions...> dims,
                  const std::vector<size_t>::const_iterator& begin,
                  const std::vector<size_t>::const_iterator& end) {
  typename removeFirst<dimensions...>::type innerDims;
  const 	size_t i      = *begin;
  constexpr size_t stride = computeProduct(innerDims);
  constexpr size_t rest   = computeOffset(innerDims, begin+1, end);
  return i * stride + rest;
}

template <int... dimensions> inline
int computeOffset(const seq<> &dims,
                  const std::vector<size_t>::const_iterator& begin,
                  const std::vector<size_t>::const_iterator& end) {
  return 0;
}

inline constexpr
int computeOffset(seq<> dims, const std::vector<size_t>& indices) {
  return 0;
}

}}
#endif
