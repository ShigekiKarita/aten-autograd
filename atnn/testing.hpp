/**
   \file testing.hpp
   \brief Testing utility
*/
#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <ATen/ATen.h>

namespace atnn {
    enum struct Color : int {
        reset = 0,
        bold = 1,
        fg_red      = 31,
        fg_green    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };

    std::ostream& operator<<(std::ostream& os, Color color) {
        return os << "\033[" << static_cast<int>(color) << "m";
    }
} // namespace atnn

#ifdef HAVE_BOOST // a BOOST dependent part
// BOOST_ENABLE_ASSERT_DEBUG_HANDLER is defined for the whole project
#include <boost/exception/info.hpp>
#include <boost/stacktrace.hpp>
#include <boost/format.hpp>
#include <boost/exception/diagnostic_information.hpp>

namespace boost {
    inline void assertion_failed_msg(char const* expr, char const* msg, char const* function, char const* file, int64_t line) {
        std::cerr << atnn::Color::bold << atnn::Color::fg_red
                  << "===== ATNN-Assetion failed =====\n"
                  << "Expression: " << expr << "\n"
                  << "Function:   " << function << " in " << file << "(" << line << ")\n"
                  << "Message:    " << (msg ? msg : "<none>") << "\n"
                  << "\n"
                  << atnn::Color::reset
                  << "=========== Backtrace ==========\n"
                  << boost::stacktrace::stacktrace() << '\n';
        std::abort();
    }

    inline void assertion_failed(char const* expr, char const* function, char const* file, int64_t line) {
        ::boost::assertion_failed_msg(expr, 0 /*nullptr*/, function, file, line);
    }
} // namespace boost

namespace atnn {
    typedef boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace> traced;
    template <class E>
    void throw_with_trace(const E& e) {
        throw boost::enable_error_info(e)
            << traced(boost::stacktrace::stacktrace());
    }
}

#define ATNN_ASSERT BOOST_ASSERT
#define ATNN_ASSERT_MSG BOOST_ASSERT_MSG
#define ATNN_ASSERT_EQ(a, b) do { BOOST_ASSERT_MSG((a) == (b), (boost::format("%d != %d") % (a) % (b)).str().c_str()); } while (0)
#define ATNN_ASSERT_SHAPE_EQ(a, b) do { \
        ATNN_ASSERT_MSG(atnn::shape_eq((a), (b)),                      \
                         (boost::format("%1 != %2") % atnn::to_tensor(a) % atnn::to_tensor(b)).str().c_str()); } while (0)

#else // not HAVE_BOOST

#define ATNN_ASSERT assert
#define ATNN_ASSERT_MSG(expr, msg) do { assert((expr) && (msg)); } while(0)
#define ATNN_ASSERT_EQ(a, b) do { ATNN_ASSERT((a) == (b)); } while (0)
#define ATNN_ASSERT_SHAPE_EQ(a, b) do { ATNN_ASSERT(atnn::shape_eq((a), (b))); } while (0)

namespace atnn {
    template <class E>
    void throw_with_trace(const E& e) {
        throw e;
    }
}

#endif


namespace atnn {

    bool is_empty(at::Tensor t) {
        return !t.defined() || t.dim() == 0;
    }

    static bool all_bool(at::Tensor x) {
        return at::Scalar(x.all()).toByte();
    }

    static bool allclose(at::Tensor actual, at::Tensor desired, float rtol=1e-7, float atol=0) {
        // ATNN_ASSERT(!atnn::is_empty(actual));
        // ATNN_ASSERT(!atnn::is_empty(desired));
        auto t = ((actual - desired).abs() <= desired.abs() * rtol + atol);
        auto ok = all_bool(t);
        if (!ok) {
            auto tsum = at::Scalar(t.toType(at::kDouble).sum()).toDouble();
            std::cerr << "error: " << tsum / t.view(-1).size(0) << " ("
                      << tsum << "/" << t.view(-1).size(0) << ")" << std::endl;
        }
        return ok;
    }

    static bool all_eq(at::Tensor actual, at::Tensor desired) {
        return all_bool(actual.eq(desired));
    }

    inline static auto to_tensor(at::IntList a) {
        return CPU(at::kLong).tensorFromBlob(const_cast<int64_t*>(a.begin()), {static_cast<int64_t>(a.size())});
    }
    inline static auto to_tensor(at::Tensor a) {
        return a;
    }

    template <typename T>
    bool shape_is(T t, at::IntList shape) {
        auto a = to_tensor(t.sizes());
        auto b = to_tensor(shape);
        bool ok = a.size(0) == b.size(0) && all_bool(a == b);
        if (!ok) std::cerr << "shape does not match:\n  lhs=" << a << "\n  rh=" << b << std::endl;
        return ok;
    }

    template <typename T1, typename T2>
    bool shape_eq(T1 t1, T2 t2) {
        auto a = to_tensor(t1);
        auto b = to_tensor(t2);
        bool ok = a.size(0) == b.size(0) && all_bool(a == b);
        return ok;
    }

    template <typename... Args>
    auto list_eq(Args&&... args) {
        return shape_eq(std::forward<Args>(args)...);
    }

    template <typename Duration>
    double to_sec(const Duration& duration) {
        return 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }

    template <typename F>
    void test_common(int argc [[gnu::unused]], char** argv, F proc, bool cpu_only=false) {
        std::cout << argv[0] << std::endl; // << std::flush;
        for (auto device: {
#ifdef NO_CUDA
                at::CPU
#else
                at::CPU, at::CUDA
#endif
                    }){
#ifdef HAVE_BOOST
            try {
#endif
                if (cpu_only && device == at::CUDA) continue;
                auto start_time = std::chrono::high_resolution_clock::now();
                proc(device);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto elapsed = to_sec(end_time - start_time);
                std::cout << Color::bold << Color::fg_green << "=== PASS "
                          << at::toString(device(at::kFloat).backend())
                          << ": " << elapsed << " sec ==="
                          << Color::reset << std::endl; //  << std::flush;
#ifdef HAVE_BOOST
            } catch (const std::exception& e) {
                std::cerr << e.what() << '\n';
                const boost::stacktrace::stacktrace* st = boost::get_error_info<traced>(e);
                if (st) {
                    std::cerr << *st << '\n';
                }
                throw e;
            }
#endif
        }
    }
} // namespace atnn
