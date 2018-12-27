#include <iostream>
#include <atnn/dataset.hpp>
#include <atnn/testing.hpp>

namespace D = atnn::dataset;
using DatasetList =  D::ConcatDataset::DatasetList;


auto make_tensor_dict(long n_samples) {
    auto inputs = CPU(at::kFloat).rand({n_samples, 3, 2});
    auto targets = CPU(at::kLong).randperm(n_samples);
    D::Dataset::Item tensor_dict = {{"input", inputs}, {"target", targets}};
    return tensor_dict;
}


int main(int argc, char**argv) {
    atnn::test_common(argc, argv, [](auto device) {
            /* test tensor_dataset */
            long n_samples = 5;
            auto tensor_dict = make_tensor_dict(n_samples);;
            auto tensor_dataset = std::make_shared<D::TensorDataset>(tensor_dict);

            // re-init
            tensor_dict["input"].copy_(device(at::kFloat).rand({n_samples, 3, 2}));
            tensor_dict["target"].copy_(device(at::kLong).randperm(n_samples));

            // tensors are updated correctly
            ATNN_ASSERT_EQ(tensor_dataset->size(), n_samples);
            for (long i = 0; i < n_samples; ++i) {
                for (auto&& item : tensor_dict) {
                    ATNN_ASSERT(atnn::all_eq(tensor_dataset->get_item(i)[item.first], item.second[i]));
                }
            }

            /* test concat_dataset */
            long n_other_samples = 7;
            auto other_dict = make_tensor_dict(n_other_samples);
            auto other_dataset = std::make_shared<D::TensorDataset>(other_dict);
            DatasetList dataset_list = {tensor_dataset, other_dataset};
            auto concat_dataset = std::make_shared<D::ConcatDataset>(dataset_list);

            ATNN_ASSERT(atnn::list_eq(concat_dataset->cummulative_sizes,
                                      at::IntList {n_samples, n_samples + n_other_samples}));
            ATNN_ASSERT_EQ(concat_dataset->size(), n_samples + n_other_samples);
            for (long i = 0; i < n_samples; ++i) {
                for (auto&& item : tensor_dict) {
                    ATNN_ASSERT(atnn::all_eq(concat_dataset->get_item(i)[item.first], item.second[i]));
                }
            }
            for (long i = 0; i < n_other_samples; ++i) {
                for (auto&& item : other_dict) {
                    ATNN_ASSERT(atnn::all_eq(concat_dataset->get_item(i + n_samples)[item.first], item.second[i]));
                }
            }

            DatasetList dataset_list2 = {tensor_dataset, concat_dataset};
            auto concat_dataset2 = std::make_shared<D::ConcatDataset>(dataset_list2);
            ATNN_ASSERT_EQ(concat_dataset2->size(), n_samples * 2 + n_other_samples);
            for (long i = 0; i < n_samples; ++i) {
                for (auto&& item : tensor_dict) {
                    ATNN_ASSERT(atnn::all_eq(concat_dataset2->get_item(i)[item.first], item.second[i]));
                }
            }
            for (long i = 0; i < n_samples; ++i) {
                for (auto&& item : tensor_dict) {
                    ATNN_ASSERT(atnn::all_eq(concat_dataset2->get_item(i + n_samples)[item.first], item.second[i]));
                }
            }
            for (long i = 0; i < n_other_samples; ++i) {
                for (auto&& item : other_dict) {
                    ATNN_ASSERT(atnn::all_eq(concat_dataset2->get_item(i + n_samples * 2)[item.first], item.second[i]));
                }
            }
            
        }, true);
}
