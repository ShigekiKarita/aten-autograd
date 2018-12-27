#pragma once

#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>

/// for kaldi
#include <kaldi-io.h>
#include <kaldi-table.h>
#include <kaldi-matrix.h>
#include <table-types.h>

/// for json
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/document.h>

namespace asr_dataset {
/** Dataset processing **/

    /// short cut for easy access
    using InputReader = kaldi::RandomAccessBaseFloatMatrixReader;
    using InputReaderPtr = std::shared_ptr<InputReader>;
    using DocPtr = std::shared_ptr<rapidjson::Document>;
    using DocIter = typename rapidjson::Document::ConstMemberIterator;

    /// read string and convert to int
    int read_int(const rapidjson::Value& v) {
        ATNN_ASSERT(v.IsString());
        return std::stoi(v.GetString());
    }

    /// read 1d token-id target from json
    at::Tensor read_target(DocIter iter) {
        const auto& target = iter->value;

        ATNN_ASSERT(target.HasMember("tokenid"));
        std::istringstream iss(target["tokenid"].GetString());

        ATNN_ASSERT(target.HasMember("olen"));
        auto olen = read_int(target["olen"]);

        auto t = CPU(at::kLong).zeros(olen);
        std::string id;
        long i = 0;
        while (std::getline(iss, id, ' ')) {
            t[i] = std::stoi(id);
            ++i;
        }
        ATNN_ASSERT_EQ(i, olen);
        return t;
    }

    /// read 2d time-freq input (e.g., FBANK feature) using kaldi
    at::Tensor read_input(InputReaderPtr reader, DocIter iter) {
        // TODO assert reader has utt-id
        auto m = std::make_shared<kaldi::Matrix<float>>(reader->Value(iter->name.GetString()));
        auto x = atnn::memory::make_tensor(m);
        ATNN_ASSERT_EQ(x.size(0), read_int(iter->value["ilen"]));
        ATNN_ASSERT_EQ(x.size(1), read_int(iter->value["idim"]));
        return x;
    }

    /// Handle sample information in json to read input and target tensors
    struct Sample {
        Sample(DocPtr d, InputReaderPtr r, DocIter i)
            : doc(d), reader(r), iter(i),
              ilen(read_int(i->value["ilen"])),
              olen(read_int(i->value["olen"])),
              idim(read_int(i->value["idim"])),
              odim(read_int(i->value["odim"]))
            {}

        DocPtr doc; // keep this for iter lifetime
        InputReaderPtr reader;
        DocIter iter;
        std::int64_t ilen, olen, idim, odim;

        const rapidjson::Value& get(const char* query) const {
            ATNN_ASSERT(iter->value.HasMember(query));
            return iter->value[query];
        }

        auto key() const {
            ATNN_ASSERT(iter->name.IsString());
            return iter->name.GetString();
        }

        auto input() const {
            ATNN_ASSERT(doc);
            ATNN_ASSERT(reader);
            return read_input(reader, iter);
        }

        auto target() const {
            ATNN_ASSERT(doc);
            return read_target(iter);
        }
    };

    /// Gather input and target in sorted order by the length, and combine them into minibatch
    auto make_batchset(DocPtr doc, InputReaderPtr reader, size_t batch_size=32,
                       size_t max_length_in=800, size_t max_length_out=150,
                       size_t max_num_batches=std::numeric_limits<size_t>::max()) {
        std::vector<Sample> keys;
        // keys.reserve(data.Size());
        ATNN_ASSERT(doc->HasMember("utts"));
        auto& data = doc->FindMember("utts")->value;
        for (auto d = data.MemberBegin(); d != data.MemberEnd(); ++d) {
            keys.emplace_back(doc, reader, d);
        }
        // shorter first
        std::sort(keys.begin(), keys.end(),
                  [](const auto& a, const auto& b) {
                      return a.olen < b.olen;  });
        std::vector<std::vector<Sample>> batchset;
        size_t start_id = 0;
        while (true) {
            const auto& start = keys[start_id];
            auto factor = std::max<size_t>(start.ilen / max_length_in, start.olen / max_length_out);
            auto b = std::max<size_t>(1, batch_size / (1 + factor));
            auto end_id = std::min<size_t>(keys.size(), start_id + b);
            std::vector<Sample> mb(keys.begin() + start_id, keys.begin() + end_id);
            batchset.push_back(mb);

            if (end_id == keys.size() || batchset.size() > max_num_batches) break;
            start_id = end_id;
        }
        return batchset;
    }

    /// Keep tensor unique_ptr for input/target as two padded tensors in a minibatch
    struct MiniBatch {
        MiniBatch(const std::vector<Sample>& minibatch) {
            this->input_lengths.reserve(minibatch.size());
            this->target_lengths.reserve(minibatch.size());
            std::int64_t max_ilen = 0, max_olen = 0;
            for (const auto& sample : minibatch) {
                this->input_lengths.push_back(sample.ilen);
                this->target_lengths.push_back(sample.olen);
                if (sample.ilen > max_ilen) max_ilen = sample.ilen;
                if (sample.olen > max_olen) max_olen = sample.olen;
            }
            auto mb_size = static_cast<std::int64_t>(minibatch.size());
            // maybe the bug of at::Tensor, memory leaks unless this unique_ptr idiom
            // TODO make Variable().data unique_ptr
            this->inputs = std::make_unique<at::Tensor>(at::CPU(at::kFloat).zeros({mb_size, max_ilen, minibatch.front().idim}));
            this->targets = std::make_unique<at::Tensor>(at::CPU(at::kLong).zeros({mb_size, max_olen}));
            for (size_t batch_idx = 0; batch_idx < minibatch.size(); ++batch_idx) {
                auto x = minibatch[batch_idx].input();
                auto t = minibatch[batch_idx].target();
                (*inputs)[batch_idx].slice(0, 0, x.size(0)) = x;
                (*targets)[batch_idx].slice(0, 0, t.size(0)) = t;
            }
        }

        std::unique_ptr<at::Tensor> inputs, targets;
        std::vector<std::int64_t> input_lengths, target_lengths;
    };

    /// read a json from a filename
    auto read_json(const std::string& filename) {
        auto doc = std::make_shared<rapidjson::Document>();
        std::ifstream ifs(filename);
        rapidjson::IStreamWrapper isw(ifs);
        doc->ParseStream(isw);
        return doc;
    }
} // namespace asr_dataset
