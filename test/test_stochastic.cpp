#include <atnn/function.hpp>
#include <atnn/chain.hpp>
#include <atnn/grad_check.hpp>
#include <atnn/testing.hpp>

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

    atnn::Variable v1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << v1 << std::endl;
    atnn::Variable v2 = {{1.0f, 2.0f},
                         {3.0f, 4.0f},
                         {5.0f, 6.0f}};
    std::cout << v2 << std::endl;
}
