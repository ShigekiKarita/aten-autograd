/**
 * \file dataloader.hpp
 * \brief Data-loader utility
 * \example test_dataloader.cpp
 */
#pragma once

#include <iterator>
#include <memory>

#include "cxx14_thread_pool/thread_pool.hpp"
#include "dataset.hpp"
#include "sampler.hpp"

namespace atnn {
    /**
        \brief Data-loader utility
        \ingroup atnn::dataloader
     */
    namespace dataloader {
        template <typename Shape>
        auto batch_shape(int64_t batch_size, Shape&& shape) {
            std::vector<int64_t> sizes = {batch_size};
            sizes.reserve(shape.size());
            std::copy(shape.begin(), shape.end(), std::back_inserter(sizes));
            return sizes;
        }

        struct DefaultCollate {
            // TODO: write more like https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L83
            at::Tensor operator()(const std::vector<at::Tensor>& batch) const {
                auto ret = batch[0].type().zeros(batch_shape(batch.size(), batch[0].sizes()));
                for (size_t i = 0; i < batch.size(); ++i) {
                    ret[i] = batch[i];
                }
                return ret;
            }
        };

        template <typename Sampler = sampler::SequentialSampler,
                  typename Collater = DefaultCollate>
        struct DataLoader {
            std::shared_ptr<dataset::Dataset> dataset;
            int64_t batch_size = 1;
            int64_t num_workers = 1;
            bool pin_memory = false;
            bool drop_last = false;
            Collater collate_fn;
            sampler::BatchSampler sampler;

            DataLoader(std::shared_ptr<dataset::Dataset> dataset,
                       int64_t batch_size=1, int64_t num_workers=1, Collater collater=DefaultCollate(),
                       bool pin_memory=false, bool drop_last=false)
                : dataset(dataset)
                , batch_size(batch_size)
                , num_workers(num_workers)
                , pin_memory(pin_memory)
                , drop_last(drop_last)
                , collate_fn(collater)
                , sampler(std::make_unique<Sampler>(dataset), batch_size, drop_last) {
                if (num_workers <= 0) {
                    num_workers = std::max(1u, std::thread::hardware_concurrency());
                }
            }

            auto size() const { return this->sampler.size(); }

            using IndexType = sampler::Sampler::IndexType;
            using Item = dataset::Dataset::Item;

            /** wrapper iterator of Sampler object

                \todo stop using std::iterator (deprecated in C++17)
             */
            struct Iterator : public std::iterator<std::input_iterator_tag, Item> {
                DataLoader* loader;
                std::vector<int64_t> indices;

                // ctors
                Iterator(DataLoader* loader, std::vector<int64_t>&& indices) : loader(loader), indices(std::move(indices)) {}
                // Iterator(DataLoader* loader) : Iterator(loader, loader->sampler.next()) {}
                Iterator(const Iterator&) = default;
                Iterator& operator=(const Iterator&) = default;

                Item operator*() {
                    std::unordered_map<std::string, std::vector<at::Tensor>> item_list;
                    item_list.reserve(this->indices.size());
                    for (auto i : this->indices) {
                        auto items = loader->dataset->get_item(i);
                        for (auto item : items) {
                            item_list[item.first].push_back(item.second);
                        }
                    }

                    Item batch_item;
                    auto num_workers = this->loader->num_workers;
                    if (num_workers == 1) {
                        for (auto&& list : item_list) {
                            batch_item[list.first] = loader->collate_fn(list.second);
                        }
                    } else {
                        thread_pool::ThreadPool<> pool(num_workers);
                        for (auto&& list : item_list) {
                            pool.enqueue([&] { batch_item[list.first] = loader->collate_fn(list.second); });
                        }
                    }
                    return batch_item;
                }
                /// ++iter
                Iterator& operator++() {
                    this->indices = loader->sampler.next();
                    return *this;
                }
                /// iter++
                Iterator operator++(int) {
                    std::vector<int64_t> backup_indices = this->indices;
                    this->indices = loader->sampler.next();
                    return Iterator(loader, std::move(backup_indices));
                }

                bool operator==(const Iterator& b) const {
                    auto& a = *this;
                    if (a.loader != b.loader) return false;
                    else if (a.indices.data() == b.indices.data()) return true;
                    else if (a.indices.empty() && b.indices.empty()) return true; // end
                    else return false;
                }
                bool operator!=(const Iterator& b) const {
                    return !(*this == b);
                }
            };
            auto begin() {
                this->sampler.reset(); // TODO: reconsider this
                return Iterator(this, this->sampler.next());
            }

            auto end() {
                return Iterator(this, {});
            }
        };

        // template<typename Sampler=sampler::SequentialSampler,
        //          typename Collater=DefaultCollate,
        //          typename _Iter = typename DataLoader<Sampler, Collater>::Iterator>
        // bool operator==(const _Iter& a, const _Iter& b) {
        //     if (a.loader != b.loader) return false;
        //     else if (a.indices.data() == b.indices.data()) return true;
        //     else if (a.indices.empty() && b.indices.empty()) return true; // end
        //     else return false;
        // }


        // template<typename Sampler=sampler::SequentialSampler,
        //          typename Collater=DefaultCollate,
        //          typename _Iter = typename DataLoader<Sampler, Collater>::Iterator>
        // bool operator!=(const _Iter& a, const _Iter& b) {
        //     return !operator==(a, b);
        // }

    }
}
