/**

## TODO

- logging
- command line option
- load dict
- encoder-decoder implementation
- attention implementation


## Usage

before running this script, prepare dataset as follows

```
# install espnet for preprocessing
git clone https://github.com/espnet/espnet
cd espnet/tools
make -f conda.mk
cd ../egs/wsj/asr1

# run preprocessing (takes about 30 min)
./run.sh --wsj0 <your-path> --wsj1 <your-path>  # stop before training stage

# link kaldi, the feats and labels
ln -s <espnet>/tools/kaldi <this script path>/kaldi
ln -s `pwd`/dump <this script path>/wsj
cd <this script path>

# running this script
make test-wsj
 */

#include <tuple>
#include <random>

#include <kaldi-io.h>

#include <ATen/ATen.h>
#include <atnn/variable.hpp>
#include <atnn/chain.hpp>
#include <atnn/optimizer.hpp>
#include <atnn/dataset.hpp>
#include <atnn/dataloader.hpp>
#include <atnn/serializer.hpp>
#include <atnn/testing.hpp>
#include <atnn/memory.hpp>

#include "asr_dataset.hpp"

using atnn::Variable;
namespace C = atnn::chain;
namespace nn = atnn::chain::nn;
namespace F = atnn::function;
namespace D = atnn::dataset;
namespace S = atnn::sampler;
namespace L = atnn::dataloader;
namespace O = atnn::optimizer;

// using PackedSeq = std::tuple<Variable, at::IntList>;
struct PackedSeq {
    Variable pad;
    at::IntList lengths;
};

#define PARAM(name) {#name, (name)}

struct Encoder : C::Chain {
    std::int64_t n_feat, n_units;
    C::ChainPtr<nn::LSTM> lstm1, lstm2, lstm3;
    C::ChainPtr<nn::Linear> linear;

    Encoder(std::int64_t n_feat, std::int64_t n_units)
        : n_feat(n_feat), n_units(n_units)
        , lstm1(new nn::LSTM(n_feat, n_units))
        , lstm2(new nn::LSTM(n_units, n_units))
        , lstm3(new nn::LSTM(n_units, n_units))
        , linear(new nn::Linear(n_units, n_units)) {
        this->chain_dict = {
            PARAM(lstm1),
            PARAM(lstm2),
            PARAM(lstm3),
            PARAM(linear)
        };
    }

    PackedSeq forward(Variable xs, at::IntList xlen) {
        auto n_batch = xs.size(0);
        auto n_time = xs.size(1);
        std::vector<Variable> h(3);
        std::vector<Variable> c(3);
        std::vector<Variable> ys;
        ys.reserve(n_time);
        xs = xs.transpose(0, 1); // time first
        // TODO masking
        for (std::int64_t t = 0; t < n_time; ++t) {
            std::tie(h[0], c[0]) = this->lstm1(xs[t], h[0], c[0]);
            if (t % 2 == 0) {
                std::tie(h[1], c[1]) = this->lstm2(h[0], h[1], c[1]);
            }
            if (t % 4 == 0) {
                std::tie(h[2], c[2]) = this->lstm2(h[1], h[2], c[2]);
                auto y =  this->linear(h[2]);
                ys.push_back(y.view({n_batch, 1, this->n_units}));
            }
            // auto y =  this->linear(h[0]);
            // ys.push_back(y.view({n_batch, 1, this->n_units}));
        }
        auto ypad = C::cat(ys, 1);
        std::vector<std::int64_t> hlen;
        for (auto xl : xlen) {
            hlen.push_back(xl / 4);
        }
        return {ypad, hlen}; // [n_batch, max_time, n_vocab]
    }
};

struct Summation : C::Chain {
    std::int64_t dim;
    Summation(std::int64_t dim) : dim(dim) {}

    /**
       @param query: (batch, dim)
       @param processed_memory: (batch, max_time, dim)
       @return (batch, max_time)
     */
    Variable forward(Variable query, Variable processed_memory) {
        ATNN_ASSERT_EQ(query.size(0), processed_memory.size(0));
        ATNN_ASSERT_EQ(query.size(1), this->dim);
        ATNN_ASSERT_EQ(processed_memory.size(2), this->dim);
        return processed_memory.sum(2);
    }
};

/**
A.k.a Dot-product attention

- Effective Approaches to Attention-based Neural Machine Translation
  [Minh-Thang Luong, arXiv, 2015/08]
  https://arxiv.org/abs/1508.04025
*/
struct LuongAttention {
    std::int64_t dim;
    LuongAttention(std::int64_t dim) : dim(dim) {}
    // not implemented now
};

/**
a.k.a Additive attention

- Neural Machine Translation by Jointly Learning to Align and Translate
  [Dzmitry Bahdanau, sec: Kyunghyun Cho, last: Yoshua Bengio, ICLR 2015, arXiv, 2014/09]
  https://arxiv.org/abs/1409.0473

- pytorch implementation
  https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/attention.py
*/
struct BahdanauAttention : C::Chain {
    std::int64_t dim;
    C::ChainPtr<nn::Linear> query_linear, value_linear;
    BahdanauAttention(std::int64_t dim)
        : dim(dim)
        , query_linear(new nn::Linear(dim, dim, /*use_bias*/ false))
        , value_linear(new nn::Linear(dim, 1, /*use_bias*/ false)) {
        this->chain_dict = {
            PARAM(query_linear),
            PARAM(value_linear)
        };
    }
    /**
       @param query: (batch, dim)
       @param processed_memory: (batch, max_time, dim)
       @return (batch, max_time)
     */
    Variable forward(Variable query, Variable processed_memory) {
        ATNN_ASSERT_EQ(query.size(0), processed_memory.size(0));
        ATNN_ASSERT_EQ(query.size(1), this->dim);
        ATNN_ASSERT_EQ(processed_memory.size(2), this->dim);
        // (bath, 1, dim)
        auto processed_query = this->query_linear(query).unsqueeze(1);
        // (batch, max_time, dim)
        auto h = nn::tanh(processed_query + processed_memory);
        // (batch, max_time)
        auto alignment = this->value_linear(h.view({h.size(0) * h.size(1), h.size(2)}))
            .view({h.size(0), h.size(1)});
        return alignment;
    }
};

/**
Get mask from lengths
 */
auto get_mask_from_lengths(at::Tensor memory, at::IntList memory_lengths) {
    ATNN_ASSERT_EQ(memory.size(0), memory_lengths.size());
    auto mask = memory.type().zeros({memory.size(0), memory.size(1)});
    for (size_t b = 0; b < memory_lengths.size(); ++b) {
        mask[b].slice(0, 0, memory_lengths[b]) = 1;
    }
    return mask;
}

void test_attention() {
    std::vector<std::int64_t> lens = {1, 3, 2};
    // auto t = Variable(at::CPU(at::kLong).zeros(3, 5, 2));
    Variable input = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    Variable expect = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    auto mask = get_mask_from_lengths(input.data(), lens);
    auto masked = input * mask;
    std::cout << masked << std::endl;
    // auto expected = 
    // ATNN_ASSERT
    // ATNN_ASSERT(atnn::all_eq(t, mask));
}


struct AttentionStates {
    Variable h,    // (batch, n_output)
        c,         // (batch, n_output)
        alignment, // (batch, max_time)
        attention; // (batch, n_output)
};

template <typename Attention>
struct AttentionLSTM : C::Chain {
    C::ChainPtr<Attention> attention_mechanism;
    C::ChainPtr<nn::LSTM> rnn_cell;
    double score_mask_value;
    std::int64_t n_input, n_output;
    AttentionLSTM(std::int64_t n_input, std::int64_t n_output, 
                  double score_mask_value = std::numeric_limits<double>::min())
        : attention_mechanism(new Attention(n_output))
        , rnn_cell(new nn::LSTM(n_input + n_output, n_output))
        , score_mask_value(score_mask_value)
        , n_input(n_input), n_output(n_output) {
        this->chain_dict = {
            PARAM(attention_mechanism),
            PARAM(rnn_cell)
        };
    }

    /**
    @param query : (batch, n_input)
    @param memory : (batch, max_time, n_output)
    @param states : see AttentionStates
    @param processed_memory
    @param memory_lengths : list of lengths before padded
     */
    auto forward(Variable query, const AttentionStates& states,
                 Variable memory, Variable processed_memory, Variable mask) {
        AttentionStates ret;
        auto att = states.attention.defined() 
            ? states.attention
            : query.data().type().zeros({query.size(0), this->n_output});

        auto rnn_input = C::cat({query, att}, -1);
        std::tie(ret.h, ret.c) = rnn_cell(rnn_input, states.h, states.c);
        //  (batch, max_time)
        auto alignment = this->attention_mechanism(ret.h, processed_memory);
        if (mask.defined()) {
            auto m = mask.data().view(alignment.sizes());
            alignment = m * alignment + (1.0 - m) * this->score_mask_value;
        }
        // normalize
        ret.alignment = nn::softmax(alignment);
        // TODO implement variable bmm
        std::vector<Variable> attention_list;
        auto n_batch = query.size(0);
        attention_list.reserve(n_batch);
        for (std::int64_t b = 0; b < n_batch; ++b) {
            // (1, max_time) x (max_time, n_output)
            attention_list.push_back(nn::mm(ret.alignment[b].unsqueeze(0), memory[b]));
        }
        // (batch, n_output)
        ret.attention = C::cat(attention_list, 0);
        return ret;
    }
};

struct Decoder : C::Chain {
    std::int64_t n_vocab, n_embed, n_units;
    C::ChainPtr<nn::Embedding> embed;
    using Att = AttentionLSTM<BahdanauAttention>;
    C::ChainPtr<Att> lstm1; //, lstm2;
    C::ChainPtr<nn::Linear> linear, memory_linear;

    Decoder(std::int64_t n_vocab, std::int64_t n_embed, std::int64_t n_units)
        : n_vocab(n_vocab), n_embed(n_embed), n_units(n_units)
        , embed(new nn::Embedding(n_vocab, n_embed))
        , lstm1(new Att(n_embed, n_units))
        // , lstm2(new nn::LSTM(n_units, n_units))
        , linear(new nn::Linear(n_units, n_vocab))
        , memory_linear(new nn::Linear(n_units, n_units, false)) {
        this->chain_dict = {
            PARAM(embed),
            PARAM(lstm1),
            // PARAM(lstm2),
            PARAM(linear),
            PARAM(memory_linear)
        };
    }

    // forced decoding
    PackedSeq forward(Variable ts, at::IntList tlen, Variable hpad, at::IntList hlen) {
        auto max_memory_time = hpad.size(1);
        auto n_time = ts.size(1);
        auto n_batch = ts.size(0);

        Variable mask = get_mask_from_lengths(hpad.data(), hlen).unsqueeze(2);
        auto pre = mask * this->memory_linear(hpad.view({n_batch * max_memory_time, hpad.size(2)}))
            .view({n_batch, max_memory_time, this->n_units});
        auto es = this->embed(ts.view({n_batch * n_time}))
            .view({n_batch, n_time, this->n_embed})
            .transpose(0, 1); // time first
        std::vector<Variable> ys;
        ys.reserve(n_time);
        AttentionStates s;
        for (std::int64_t t = 0; t < n_time; ++t) {
            s = this->lstm1->forward(es[t], s, hpad, pre, mask);
            auto y = this->linear(s.h);
            ys.push_back(y.view({n_batch, 1, n_vocab}, "decoder-y"));
        }
        // TODO masking
        auto ypad = C::cat(ys, 1); // [n_batch, n_time, n_vocab]
        return {ypad, tlen};
    }
};


struct Model : C::Chain {
    std::int64_t n_feat, n_vocab, n_embed, n_units;
    C::ChainPtr<Encoder> encoder;
    C::ChainPtr<Decoder> decoder;

    Model(std::int64_t n_feat, std::int64_t n_vocab,
          std::int64_t n_embed=200, std::int64_t n_units=200)
        : n_feat(n_feat), n_vocab(n_vocab),
          n_embed(n_embed), n_units(n_units),
          encoder(new Encoder(n_feat, n_units)),
          decoder(new Decoder(n_vocab, n_embed, n_units)) {
        this->chain_dict = {
            PARAM(encoder),
            PARAM(decoder)
        };
    }

     auto forward(Variable xs, at::IntList xlen,
                  Variable ts, at::IntList tlen) {
        auto n_batch = xs.size(0);
        auto [hs, hlen] = this->encoder->forward(xs, xlen);
        auto [ys, ylen] = this->decoder->forward(ts, tlen, hs, hlen);
        auto loss = nn::cross_entropy(ys.contiguous().view({n_batch * ys.size(1), ys.size(2)}),
                                      ts.contiguous().view({-1}),
                                      /*size_average*/ false, /*ignore_index*/ 0) / n_batch;
        return loss;
    }
};

int main() {
    test_attention();
    // testing float-vector io
    std::string train_dir = "wsj/dump/train_si84/deltafalse";
    std::string dev_dir = "wsj/dump/test_dev93/deltafalse";

    std::cout << "loading json" << std::endl;
    auto train_json = asr_dataset::read_json(train_dir + "/data.json");
    auto dev_json = asr_dataset::read_json(dev_dir + "/data.json");

    // see http://kaldi-asr.org/doc/classkaldi_1_1RandomAccessTableReader.html
    std::cout << "making batchset" << std::endl;
    auto train_reader = std::make_shared<kaldi::RandomAccessBaseFloatMatrixReader>("scp:" + train_dir + "/feats.scp");
    auto dev_reader = std::make_shared<kaldi::RandomAccessBaseFloatMatrixReader>("scp:" + dev_dir + "/feats.scp");
    auto train_batchset = asr_dataset::make_batchset(train_json, train_reader);
    auto dev_batchset = asr_dataset::make_batchset(dev_json, dev_reader);

    std::cout << "the longest (input) data information" << std::endl;
    auto last = dev_batchset.back().back();
    std::cout << "input: " << last.input().sizes() << std::endl;
    std::cout << "target: " << last.target().sizes() << std::endl;
    auto n_vocab = asr_dataset::read_int(last.get("odim"));
    auto n_feat = asr_dataset::read_int(last.get("idim"));

    // set seed
    int seed = 0;
    std::mt19937 engine(seed);

    auto device = at::CUDA;
    Model model(n_feat, n_vocab, 320, 320);
    if (device == at::CUDA) {
        model.toBackend(at::kCUDA);
    }
    auto optimizer = O::Adadelta(model.parameters());

    std::shuffle(train_batchset.begin(), train_batchset.end(), engine);
    for (bool is_train : {true, false}) {
        const auto& batchset = is_train ? train_batchset : dev_batchset;
        for (const auto& samples : batchset) {
            asr_dataset::MiniBatch minibatch(samples);
            if (is_train) {
                Variable xpad = *(minibatch.inputs);
                Variable tpad = *(minibatch.targets);
                if (device == at::CUDA) {
                    xpad.toBackend(at::kCUDA);
                    tpad.toBackend(at::kCUDA);
                }

                auto loss = model.forward(xpad, minibatch.input_lengths,
                                          tpad, minibatch.target_lengths);
                std::cout << "loss: " << loss << std::endl;
                loss.clear_grads();
                loss.backward();
                loss.detach();
                auto gnorm = optimizer.clip_grad_norm(0.25);
                std::cout << "grad-norm: " << gnorm << std::endl;
                optimizer.update();
            }
        }
    }
}
