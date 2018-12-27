#include <iostream>
#include <atnn/sampler.hpp>
#include <atnn/dataset.hpp>
#include <atnn/dataloader.hpp>
#include <atnn/testing.hpp>

namespace D = atnn::dataset;
namespace S = atnn::sampler;
namespace L = atnn::dataloader;

auto make_tensor_dict(long n_samples) {
    auto inputs = CPU(at::kFloat).rand({n_samples, 3});
    auto targets = CPU(at::kLong).arange(n_samples);
    D::Dataset::Item tensor_dict = {{"input", inputs}, {"target", targets}};
    return tensor_dict;
}


int main(int argc, char**argv) {
    atnn::test_common(argc, argv, []([[gnu::unused]] auto device) {
            /* test tensor_dataset */
            long n_samples = 10;
            auto tensor_dict = make_tensor_dict(n_samples);;
            auto tensor_dataset = std::make_shared<D::TensorDataset>(tensor_dict);
            {
                auto loader = L::DataLoader<S::SequentialSampler>(tensor_dataset, 3, 4);
                ATNN_ASSERT_EQ(loader.size(), 4);
                for (auto iter = loader.begin(); iter != loader.end(); iter++) {
                    auto item = *iter;
                    ATNN_ASSERT(atnn::list_eq(item["target"], iter.indices));
                }

                for (auto item : loader) {
                    // std::cout << "target:" << item["target"] << std::endl;
                    // std::cout << item["input"] << std::endl;
                }
            }
            {
                auto loader = L::DataLoader<S::RandomSampler>(tensor_dataset, 3, 4);
                ATNN_ASSERT_EQ(loader.size(), 4);
                for (auto iter = loader.begin(); iter != loader.end(); iter++) {
                    auto item = *iter;
                    ATNN_ASSERT(atnn::list_eq(item["target"], iter.indices));
                }

                for (auto item : loader) {
                    // std::cout << "target:" << item["target"] << std::endl;
                    // std::cout << item["input"] << std::endl;
                }
            }
        }, true);
}
