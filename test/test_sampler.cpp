#include <iostream>
#include <atnn/sampler.hpp>
#include <atnn/dataset.hpp>
#include <atnn/testing.hpp>

namespace D = atnn::dataset;
namespace S = atnn::sampler;
using DatasetList =  D::ConcatDataset::DatasetList;


auto make_tensor_dict(long n_samples) {
    auto inputs = CPU(at::kFloat).rand({n_samples, 3, 2});
    auto targets = CPU(at::kLong).randperm(n_samples);
    D::Dataset::Item tensor_dict = {{"input", inputs}, {"target", targets}};
    return tensor_dict;
}


int main(int argc, char**argv) {
    atnn::test_common(argc, argv, []([[gnu::unused]] auto device) {
            /* test tensor_dataset */
            long n_samples = 10;
            auto tensor_dict = make_tensor_dict(n_samples);;
            auto tensor_dataset = std::make_shared<D::TensorDataset>(tensor_dict);
            S::SequentialSampler seq_sampler(tensor_dataset);
            S::RandomSampler rand_sampler(tensor_dataset);
            std::vector<S::Sampler::IndexType> rands;
            rands.reserve(n_samples);
            for (long i = 0; i < n_samples; ++i) {
                ATNN_ASSERT_EQ(seq_sampler.remain(), n_samples - i);
                ATNN_ASSERT_EQ(rand_sampler.remain(), n_samples - i);
                ATNN_ASSERT_EQ(seq_sampler.next(), i);
                rands.push_back(rand_sampler.next());
            }
            std::sort(rands.begin(), rands.end());
            for (long i = 0; i < n_samples; ++i) {
                ATNN_ASSERT_EQ(rands[i], i);
            }

            long n_batch = 3;
            {
                S::BatchSampler batch_sampler(std::make_unique<S::SequentialSampler>(tensor_dataset), n_batch);
                ATNN_ASSERT_EQ(batch_sampler.remain(), 4);
                long i = 0;
                while (batch_sampler.remain() > 0) {
                    auto batch = batch_sampler.next();
                    if (batch_sampler.remain() == 0) {
                        ATNN_ASSERT(atnn::list_eq(batch, at::IntList {9}));
                    } else {
                        ATNN_ASSERT(atnn::list_eq(batch, at::IntList {i, i + 1, i + 2}));
                    }
                    i += n_batch;
                }
            }
            {
                S::BatchSampler batch_sampler(std::make_unique<S::SequentialSampler>(tensor_dataset), n_batch, true);
                ATNN_ASSERT_EQ(batch_sampler.remain(), 3);
                long i = 0;
                while (batch_sampler.remain() > 0) {
                    auto batch = batch_sampler.next();
                    ATNN_ASSERT(atnn::list_eq(batch, at::IntList {i, i + 1, i + 2}));
                    i += n_batch;
                }

            }
        }, true);
}
