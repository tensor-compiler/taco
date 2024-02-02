#ifndef TACO_UTIL_BENCHMARK_H
#define TACO_UTIL_BENCHMARK_H

#include <chrono>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>
#include "taco/error.h"

using namespace std;

namespace taco {
namespace util {

struct TimeResults {
  double mean;
  double stdev;
  double median;
  int size;

  friend std::ostream& operator<<(std::ostream& os, const TimeResults& t) {
    if (t.size == 1) {
      return os << t.mean;
    }
    else {
      return os << "  mean:   " << t.mean   << endl
                << "  stdev:  " << t.stdev  << endl
                << "  median: " << t.median;
    }
  }
};

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

/// Monotonic timer that can be called multiple times and that computes
/// statistics such as mean and median from the calls.
class Timer {
public:
  void start() {
    begin = std::chrono::steady_clock::now();
    paused = false;
  }

  void pause() {
    if (paused) {
        assert(false);
    }
    auto paused_time = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration<double, std::milli>(paused_time - begin).count();
    paused_times.push_back(diff);
    paused = true;
  }

  void stop() {
    if (paused) {
		assert(false);
	}
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();

    if (!paused_times.empty()) {
	for (auto ptimes : paused_times) {
	    diff += ptimes; 
	}
	paused_times.clear();
    }

    times.push_back(diff);
  }

  // Compute mean, standard deviation and median
  TimeResults getResult() {
    int repeat = static_cast<int>(times.size());

    TimeResults result;
    double mean=0.0;
    // times = ends - begins
    sort(times.begin(), times.end());
    // remove 10% worst and best cases
    const int truncate = static_cast<int>(repeat * 0.1);
    mean = accumulate(times.begin() + truncate, times.end() - truncate, 0.0);
    int size = repeat - 2 * truncate;
    result.size = size;
    mean = mean/size;
    result.mean = mean;

    vector<double> diff(size);
    transform(times.begin() + truncate, times.end() - truncate,
              diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = inner_product(diff.begin(), diff.end(),
                                  diff.begin(), 0.0);
    result.stdev = std::sqrt(sq_sum / size);
    result.median = (size % 2)
                    ? times[size/2]
                    : (times[size/2-1] + times[size/2]) / 2;
    return result;
  }

  double clear_cache() {
    double ret = 0.0;
    if (!dummyA) {
      dummyA = (double*)(malloc(dummySize*sizeof(double)));
      dummyB = (double*)(malloc(dummySize*sizeof(double)));
    }
    for (int i=0; i< 100; i++) {
      dummyA[rand() % dummySize] = rand()/RAND_MAX;
      dummyB[rand() % dummySize] = rand()/RAND_MAX;
    }
    for (int i=0; i<dummySize; i++) {
      ret += dummyA[i] * dummyB[i];
    }
    return ret;
  }

protected:
  vector<double> times;
  TimePoint begin;
  vector<double> paused_times;
  bool paused = false;
private:
  int dummySize = 3000000;
  double* dummyA = NULL;
  double* dummyB = NULL;
};


/// Monotonic timer that prints results as it goes.
class LapTimer {
public:
  LapTimer(string timerName = "") : timerGroup(true), isTiming(false) {
    if (timerName != "") {
      std::cout << timerName << std::endl;
    }
  }

  void start(const string& name) {
    this->timingName = name;
    taco_iassert(!isTiming) << "Called PrintTimer::start twice in a row";
    isTiming = true;
    begin = std::chrono::steady_clock::now();
  }

  void lap(const string& name) {
    auto end = std::chrono::steady_clock::now();
    taco_iassert(isTiming) << "lap timer that hasn't been started";
    if (timerGroup) {
      std::cout << "  ";
    }
    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << timingName << ": " << diff << " ms" << std::endl;

    this->timingName = name;
    begin = std::chrono::steady_clock::now();
  }

  void stop() {
    auto end = std::chrono::steady_clock::now();
    taco_iassert(isTiming)
        << "Called PrintTimer::stop without first calling start";
    if (timerGroup) {
      std::cout << "  ";
    }
    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << timingName << ": " << diff << " ms" << std::endl;
    isTiming = false;
  }

private:
  bool timerGroup;
  string timingName;
  TimePoint begin;
  bool isTiming;
};

}}

#define TACO_TIME_REPEAT(CODE, REPEAT, RES, COLD) {  \
    taco::util::Timer timer;                         \
    for(int i=0; i<REPEAT; i++) {                    \
      if(COLD)                                       \
        timer.clear_cache();                         \
      timer.start();                                 \
      CODE;                                          \
      timer.stop();                                  \
    }                                                \
    RES = timer.getResult();                         \
  }

#endif
