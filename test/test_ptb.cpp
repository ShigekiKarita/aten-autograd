#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <ATen/ATen.h>
#include <atnn/chain.hpp>
#include <atnn/optimizer.hpp>
#include <atnn/dataset.hpp>
#include <atnn/dataloader.hpp>
#include <atnn/testing.hpp>

namespace C = atnn::chain;
namespace nn = atnn::chain::nn;
namespace F = atnn::function;
namespace D = atnn::dataset;
namespace S = atnn::sampler;
namespace L = atnn::dataloader;
namespace O = atnn::optimizer;

using atnn::Variable;

static const std::string eos = "<eos>";

struct Dictionary {
    std::vector<std::string> idx2word = {eos};
    std::unordered_map<std::string, long> word2idx = {{eos, 0}};

    auto add_word(const std::string& word) {
        if (word2idx.count(word) == 0) {
            this->idx2word.push_back(word);
            this->word2idx[word] = this->idx2word.size() - 1;
        }
        return this->word2idx[word];
    }

    auto size() { return this->idx2word.size(); }
};


struct Corpus {
    Dictionary dictionary;
    at::Tensor train, valid, test;

    Corpus(const std::string& path)
        : dictionary()
        , train(this->tokenize(path + "/train.txt"))
        , valid(this->tokenize(path + "/valid.txt"))
        , test(this->tokenize(path + "/test.txt")) {}

    template <typename F1, typename F2>
    auto iterate_words(const std::string& path, F1&& func1, F2&& func2) {
        long n_tokens = 0;
        std::fstream ifs(path);
        ATNN_ASSERT(ifs.is_open());
        std::string line;
        while (std::getline(ifs, line)) {
            std::stringstream ss(line);
            std::string word;
            while (std::getline(ss, word, ' ')) {
                func1(word);
            }
            func2();
        }
        return n_tokens;
    }

    auto update_dict(const std::string& path) {
        long n_tokens = 0;
        this->iterate_words(
            path,
            [&n_tokens, this](const auto& word) { this->dictionary.add_word(word); ++n_tokens; },
            [&n_tokens]() { ++n_tokens; });
        return n_tokens;
    }

    at::Tensor tokenize(const std::string& path) {
        auto n_tokens = this->update_dict(path);
        auto ids = at::CPU(at::kLong).zeros({n_tokens});
        long token = 0;
        this->iterate_words(
            path,
            [&ids, &token, this](const auto& word) { ids[token] = this->dictionary.word2idx[word]; ++token; },
            [&ids, &token, this]() { ids[token] = this->dictionary.word2idx[eos]; ++token; });
        return ids;
    }
};

auto batchify(const at::Tensor& data, long bsz) {
    auto nbatch = data.size(0) / bsz;
    auto ndata = data.narrow(0, 0, nbatch * bsz);
    return ndata.view({bsz, -1}).t().contiguous();
}

#define PARAM(name) {#name, (name)}

/// \todo implement Dropout
struct RNNLM : C::Chain {
    long n_vocab, n_embed, n_units;
    C::ChainPtr<nn::Embedding> embed;
    C::ChainPtr<nn::LSTM> lstm1, lstm2;
    C::ChainPtr<nn::Linear> linear;

    atnn::Variable h1, c1, h2, c2;

    RNNLM(long n_vocab, long n_embed=200, long n_units=200)
        : n_vocab(n_vocab), n_embed(n_embed), n_units(n_units)
        , embed(new nn::Embedding(n_vocab, n_embed))
        , lstm1(new nn::LSTM(n_embed, n_units))
        , lstm2(new nn::LSTM(n_units, n_units))
        , linear(new nn::Linear(n_units, n_vocab)) {
        this->chain_dict = {
            PARAM(embed),
            PARAM(lstm1),
            PARAM(lstm2),
            PARAM(linear)
        };
    }

    auto operator()(atnn::Variable x) {
        auto n_time = x.size(0);
        auto n_batch = x.size(1);
        auto es = this->embed(x.view({-1})).view({n_time, n_batch, this->n_embed});
        std::vector<atnn::Variable> ys;
        ys.reserve(n_time);
        for (long t = 0; t < n_time; ++t) {
            std::tie(this->h1, this->c1) = this->lstm1(es[t], this->h1, this->c1);
            std::tie(this->h2, this->c2) = this->lstm2(this->h1, this->h2, this->c2);
            auto y = this->linear(this->h2);
            ys.push_back(y.view({1, n_batch, n_vocab}));
        }
        return C::cat(ys, 0); // [n_time, n_batch, n_vocab]
    }

    void reset_state() {
        this->h1 = Variable({}, this->train);
        this->c1 = Variable({}, this->train);
        this->h2 = Variable({}, this->train);
        this->c2 = Variable({}, this->train);
    }

    void unchain_grads() {
        this->h1.ptr->grad = {};
        this->c1.ptr->grad = {};
        this->h2.ptr->grad = {};
        this->c2.ptr->grad = {};
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
            if (device == at::CPU) return;

            Corpus corpus("./data"); // cd data; ./download_ptb.sh
            long n_batch = 20, n_bptt = 35, log_interval = 200;
            auto train_tensor = batchify(corpus.train, n_batch);
            auto valid_tensor = batchify(corpus.valid, n_batch);
            auto n_vocab = static_cast<long>(corpus.dictionary.size());
            RNNLM net(n_vocab);
            if (device == at::CUDA) {
                net.toBackend(at::kCUDA);
            }

            auto optimizer = O::SGD(net.parameters(), 20.0);
            for (long epoch = 0; epoch < 40; ++epoch) {
                for (bool train : {true, false}) {
                    net.set_train(train);
                    net.reset_state();
                    auto dataset = train ? train_tensor : valid_tensor;
                    double sum_loss = 0.0;
                    long sum_item = 0;
                    auto start_time = std::chrono::high_resolution_clock::now();
                    for (long i = 0; i < dataset.size(0)-1; i += n_bptt) {
                        auto seq_len = std::min(n_bptt, dataset.size(0) - 1 - i);
                        auto xs = atnn::Variable(dataset.narrow(0, i, seq_len), train);
                        auto ts = atnn::Variable(dataset.narrow(0, i+1, seq_len).view(-1), train);
                        if (device == at::CUDA) {
                            xs.toBackend(at::kCUDA);
                            ts.toBackend(at::kCUDA);
                        }
                        auto ys = net(xs).contiguous();
                        auto loss = nn::cross_entropy(ys.view({n_batch * seq_len, n_vocab}), ts.view({n_batch * seq_len}));
                        sum_loss += at::Scalar(loss.data()).toDouble() * n_batch;
                        sum_item += n_batch;

                        if (sum_item % (log_interval * n_batch) == 0) {
                            auto elapsed = atnn::to_sec(std::chrono::high_resolution_clock::now() - start_time);
                            std::cout << (train ? "train" : "test")
                            << " iter: " << i << "/" << dataset.size(0)
                            << " loss: " << sum_loss / sum_item
                            << " ms/batch: " << 1e3 * elapsed / sum_item * n_batch << std::endl;
                        }

                        if (train) {
                            net.clear_grads();
                            loss.backward();
                            loss.detach();
                            // net.unchain_grads();
                            optimizer.clip_grad_norm(0.25);
                            optimizer.update();
                        }
                    }
                    auto elapsed = atnn::to_sec(std::chrono::high_resolution_clock::now() - start_time);
                    std::cout << (train ? "train" : "test") << "-epoch: " << epoch
                        << " loss: " << sum_loss / sum_item
                        << " ms/batch: " << 1e3 * elapsed / sum_item * n_batch << std::endl;
                }
            }
        });
}
