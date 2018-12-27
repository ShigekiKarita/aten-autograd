/**
   \file dataset.hpp
   \brief Dataset utility for atnn::dataloader::DataLoader in dataloader.hpp
   \example test_dataset.cpp
*/
#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <memory>

#include <ATen/ATen.h>
#include "testing.hpp"


namespace atnn {
    /**
        \brief Dataset utility for atnn::dataloader::DataLoader in dataloader.hpp
        \ingroup atnn::dataset
     */
    namespace dataset {
        struct Dataset {
            using Item = std::unordered_map<std::string, at::Tensor>;
            using IndexType = int64_t;

            virtual ~Dataset() {}

            auto operator[](IndexType index) const {
                if (index < 0) {
                    index += this->size();
                }
                ATNN_ASSERT(0 <= index && index < this->size());
                return this->get_item(index);
            }

            virtual Item get_item(IndexType index) const = 0;

            virtual int64_t size() const = 0;
        };


        struct TensorDataset : Dataset {
            Item tensor_dict;

            TensorDataset(Item tensor_dict) : tensor_dict(tensor_dict) {
                ATNN_ASSERT(!tensor_dict.empty());
                auto len = tensor_dict.begin()->second.size(0);
                for (auto&& item : tensor_dict) {
                    // TODO better error message to indicate item.first (key)
                    ATNN_ASSERT_EQ(item.second.size(0), len);
                }

            }

            virtual Item get_item(IndexType index) const override {
                Item item;
                for (auto tensor : this->tensor_dict) {
                    item[tensor.first] = tensor.second[index];
                }
                return item;
            }

            virtual int64_t size() const override {
                return tensor_dict.begin()->second.size(0);
            }
        };

        struct ConcatDataset : Dataset {
            using DatasetList = std::vector<std::shared_ptr<Dataset>>;
            DatasetList datasets;

            using Size = decltype(datasets[0]->size());
            std::vector<Size> cummulative_sizes;

            static auto cumsum(const DatasetList& ds) noexcept {
                std::vector<Size> ret;
                ret.reserve(ds.size());
                Size start = 0;
                for (auto d : ds) {
                    auto len = d->size();
                    ret.push_back(start + len);
                    start += len;
                }
                return ret;
            }

            ConcatDataset(DatasetList datasets)
                : datasets(datasets)
                , cummulative_sizes(cumsum(datasets)) {
                ATNN_ASSERT(!datasets.empty());
            }

            virtual Item get_item(IndexType index) const override {
                // iterator of the next cumsum-size bound to index
                auto iter = std::lower_bound(this->cummulative_sizes.cbegin(),
                                             this->cummulative_sizes.cend(),
                                             index,
                                             std::less_equal<>());
                auto d = datasets[iter - this->cummulative_sizes.begin()];
                return d->get_item(d->size() - (*iter - index));
            }

            virtual int64_t size() const override {
                return this->cummulative_sizes.back();
            }
        };

    } // namespace dataset
} // namespace atnn
