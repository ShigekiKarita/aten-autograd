#include <atnn/function.hpp>
#include <atnn/chain.hpp>
#include <atnn/grad_check.hpp>
#include <atnn/testing.hpp>


template <typename T>
using IL = std::initializer_list<T>;

int main() {
    std::initializer_list<std::initializer_list<float>> il = {{1.0f, 2.0f},
                                                              {3.0f, 4.0f},
                                                              {5.0f, 6.0f}};
    static_assert(std::is_same<float, atnn::detail::DeepElementTypeof<decltype(il)>>::value);
    at::IntList shape = {3, 2};
    auto s = atnn::detail::nested_init_list_shape(il);
    ATNN_ASSERT_SHAPE_EQ(s, shape);

    std::vector<float> v = {1, 2, 3, 4, 5, 6};
    std::vector<float> f;
    atnn::detail::flatten_list(il, f);
    for (size_t i = 0; i < v.size(); ++i) {
        ATNN_ASSERT_EQ(v[i], f[i]);
    }

    atnn::Variable v1 =
#ifdef __clang__ // clang cannot deduce this initializer_list<float>
        atnn::Init<float>
#endif
        {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << v1 << std::endl;

    atnn::Variable v2 =
#ifdef __clang__
        atnn::Init<float, 2>
#endif
        {{1.0f, 2.0f},
         {3.0f, 4.0f},
         {5.0f, 6.0f}};
    std::cout << v2 << std::endl;

    atnn::Variable v3 =
#ifdef __clang__
        atnn::Init<float, 3>
#endif
        {{{1l, 2l}, {1l, 2l}},
         {{3l, 4l}, {1l, 2l}},
         {{5l, 6l}, {1l, 2l}}};
    std::cout << v3 << std::endl;

    // generalized initializer without type deduction of initializer_list
    atnn::Variable v6 = atnn::Init<float, 6> {{{{{{1}}}}}};
}
