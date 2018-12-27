#include <functional>
#include <atnn/variable.hpp>


using atnn::Variable;

int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {

    Variable v0(device(at::kFloat).rand({3, 4}));
    Variable v1(device(at::kFloat).rand({3, 4}));
    Variable v2 = v1; // assign
    Variable v2c(v1); // copy ctor
    Variable v3(device(at::kFloat).rand({3, 4}));

    // reference semantics
    (*v2.ptr->data)[0][0] = -100;  // mutate
    ATNN_ASSERT_EQ(at::Scalar(v1.data()[0][0]).toDouble(), -100);
    ATNN_ASSERT_EQ(at::Scalar(v2c.data()[0][0]).toDouble(), -100);

    // test equality
    ATNN_ASSERT(v1 == v2);
    ATNN_ASSERT(v1 == v2c);
    ATNN_ASSERT(v0 != v1);
    ATNN_ASSERT(v0 != v2);

    // test hash
    auto hash = Variable::Hash();
    ATNN_ASSERT(hash(v1) == hash(v2));
    ATNN_ASSERT(hash(v1) == hash(v2c));
    ATNN_ASSERT(hash(v0) != hash(v1));
    ATNN_ASSERT(hash(v0) != hash(v2));

    // test unordered_map<Variable>
    Variable::Set set {v0, v1};
    ATNN_ASSERT_EQ(set.count(v2), 1); // v1 == v2
    ATNN_ASSERT_EQ(set.count(v2c), 1); // v1 == v2c
    ATNN_ASSERT_EQ(set.count(v3), 0); // not found

    Variable::Map<double> map {{v0, -1}, {v1, 1}};
    ATNN_ASSERT_EQ(map[v0], -1);
    ATNN_ASSERT_EQ(map[v1], 1);
    ATNN_ASSERT_EQ(map[v2], 1); // v1 == v2
    ATNN_ASSERT_EQ(map[v2c], 1); // v1 == v2c
    ATNN_ASSERT_EQ(map.count(v3), 0); // not found

    });
}
