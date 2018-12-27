/**
   \file sampler.hpp
   \brief Sampler utility for atnn::dataloader::DataLoader in dataloader.hpp

   \example test_sampler.cpp
   \todo reimplement these sampler: SequentialSampler->counting_iterator, RandomSampler->permutation_iterator
*/
#pragma once

#include <memory>
#include <iterator>

#include <ATen/ATen.h>
#include "dataset.hpp"

namespace atnn {
    /**
        \brief Sampler utility for atnn::dataloader::DataLoader in dataloader.hpp
        \ingroup atnn::sampler
    */
    namespace sampler {

        struct Sampler {
            using IndexType = int64_t;
            virtual IndexType size() const = 0;
            virtual IndexType next() = 0;
            virtual IndexType remain() const = 0;
            virtual void reset() = 0;
        };

        struct SequentialSampler : Sampler {
            std::shared_ptr<dataset::Dataset> data_source;
            IndexType i = 0;

            SequentialSampler(std::shared_ptr<dataset::Dataset> data_source) : data_source(data_source) {}

            virtual IndexType next() override {
                return this->i++;
            }

            virtual IndexType size() const override { return this->data_source->size(); }
            virtual IndexType remain() const override { return this->size() - this->i; }
            virtual void reset() override { this->i = 0; }
        };

        struct RandomSampler : Sampler {
            std::shared_ptr<dataset::Dataset> data_source;
            at::Tensor perm;
            IndexType i = 0;

            RandomSampler(std::shared_ptr<dataset::Dataset> data_source)
                : data_source(data_source)
                , perm(at::CPU(at::kLong).randperm(data_source->size())) {}

            virtual IndexType next() override {
                return at::Scalar(this->perm[i++]).toLong();
            }

            virtual IndexType size() const override { return this->data_source->size(); }
            virtual IndexType remain() const override { return this->size() - this->i; }
            virtual void reset() override {
                this->i = 0;
                at::CPU(at::kLong).randperm_out(this->perm, this->size());
            }
        };

        struct BatchSampler {
            std::unique_ptr<Sampler> sampler;
            const Sampler::IndexType batch_size = 1;
            bool drop_last = false;

            BatchSampler(std::unique_ptr<Sampler>&& sampler,
                         Sampler::IndexType batch_size=1, bool drop_last=false)
                : sampler(std::move(sampler))
                , batch_size(batch_size)
                , drop_last(drop_last) {
                ATNN_ASSERT(batch_size > 0);
            }

            auto& reset() {
                this->sampler->reset();
                return *this;
            }

            auto calc_batch(Sampler::IndexType n) const {
                return this->drop_last
                    ? n / this->batch_size
                    : (n + this->batch_size - 1) / this->batch_size;
            }

            auto remain() const {
                return this->calc_batch(this->sampler->remain());
            }

            auto size() const {
                return this->calc_batch(this->sampler->size());
            }

            /// NOTE: this returns empty batch {} when this->sampler->remain() == 0
            auto next() {
                std::vector<Sampler::IndexType> batch;
                batch.reserve(this->batch_size);
                while (this->sampler->remain() > 0 && batch.size() < static_cast<size_t>(this->batch_size)) {
                    batch.push_back(this->sampler->next());
                }
                return batch;
            }
        };
    }
}
