#include <iostream>
#include <taco/parser/schedule_parser.h>
#include "test.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace taco::parser;

void assert_string_vectors_equal(vector<string> a, vector<string> b) {
    ASSERT_EQ(a.size(), b.size()) << "Vectors are of unequal lengths: " << a.size() << " != " << b.size();
    for(size_t i = 0; i < a.size(); i++) {
        EXPECT_EQ(a[i], b[i]) << "a[" << i << "] != b[" << i << "]: \"" << a[i] << "\" != \"" << b[i] << "\"";
    }
}

void assert_string_vector_vectors_equal(vector<vector<string>> a, vector<vector<string>> b) {
    ASSERT_EQ(a.size(), b.size()) << "Vector-vectors are of unequal lengths: " << a.size() << " != " << b.size();
    for(size_t i = 0; i < a.size(); i++) {
        assert_string_vectors_equal(a[i], b[i]);
    }
}

TEST(schedule_parser, normal_operation) {
    struct {
        string str;
        vector<vector<string>> result;
    } cases[] = {
        // basic parsing
        { "i,j,k",                  { { "i" }, { "j" }, { "k" } } },
        { "i(j,k)",                 { { "i", "j", "k" } } },
        { "i(j,k),l(m,n)",          { { "i", "j", "k" }, { "l", "m",          "n" } } },
        { "i(j,k),l(m(n,o),p)",     { { "i", "j", "k" }, { "l", "m(n,o)",     "p" } } },
        { "i(j,k),l(m(n(o(p))),q)", { { "i", "j", "k" }, { "l", "m(n(o(p)))", "q" } } },

        // whitespace
        { "i,j, k",                  { { "i" }, { "j" }, { "k" } } },
        { "i(j, k)",                 { { "i", "j", "k" } } },
        { "i(j,k), l(m,n)",          { { "i", "j", "k" }, { "l", "m",          "n" } } },
        { "i(j,k),l(m(n, o),p)",     { { "i", "j", "k" }, { "l", "m(n,o)",     "p" } } },
        { "i(j,k),l(m(n(o(p))), q)", { { "i", "j", "k" }, { "l", "m(n(o(p)))", "q" } } },

        // empty slots
        { "",              { } },
        { ",j,k",          { { "" }, { "j" }, { "k" } } },
        { "i(,k)",         { { "i", "", "k" } } },
        { "(j,k)",         { { "", "j", "k" } } },
        { "i(j,),,l(m,n)", { { "i", "j", "" }, { "" }, { "l", "m", "n" } } },

        // real scheduling directives
        { "split(i,i0,i1,16)",           { { "split", "i", "i0", "i1", "16" } } },
        { "precompute(A(i,j)*x(j),i,i)", { { "precompute", "A(i,j)*x(j)", "i", "i" } } },
        { "split(i,i0,i1,16),precompute(A(i,j)*x(j),i,i)",
                                         { { "split", "i", "i0", "i1", "16" },
                                           { "precompute", "A(i,j)*x(j)", "i", "i" } } },
    };
    for(auto test : cases) {
        auto actual = ScheduleParser(test.str);
        cout << "string \"" << test.str << "\"" << " parsed as: " << serializeParsedSchedule(actual) << endl;
        assert_string_vector_vectors_equal(test.result, actual);
    }
}

TEST(schedule_parser, error_reporting) {
    struct {
        string str;
        string assertion;
    } cases[] = {
        { "i,j,k(",  "too few right-parens" },
        { "i(j,k",   "too few right-parens" },
        { "i,j,k)",  "too many right-parens" },
        { "i,j,k)(", "too many right-parens" },
    };
    for(auto test : cases) {
        try {
            auto actual = ScheduleParser(test.str);
            // should throw an exception before getting here
            ASSERT_TRUE(false);
        } catch (taco::TacoException &e) {
            string message = e.what();
            EXPECT_TRUE(message.find(test.assertion) != string::npos)
              << "substring \"" << test.assertion << "\" not found in exception message \"" << message << "\"";
        }
    }
}
