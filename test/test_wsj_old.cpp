/**

## TODO

- minibatch 
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

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>

#include <kaldi-io.h>
#include <kaldi-table.h>
#include <kaldi-matrix.h>
#include <table-types.h>

// download from 
// https://github.com/nlohmann/json/releases/download/v3.1.1/json.hpp
#include <json.hpp>
using nlohmann::json;

#include <ATen/ATen.h>
#include <atnn/variable.hpp>
#include <atnn/memory.hpp>
#include <atnn/testing.hpp>

template <typename JsonValue>
auto get_int(const JsonValue& utt, const std::string& key="olen") {
    return std::stoi(utt[key].template get<std::string>());
}

template <typename JsonValue>
at::Tensor make_target(const JsonValue& target) {
    std::istringstream iss(target["tokenid"].template get<std::string>());
    auto olen = get_int(target, "olen");
    auto t = CPU(at::kLong).zeros(olen);
    std::string id;
    long i = 0;
    while ( std::getline(  iss, id, ' ' ) ) {
        t[i] = std::stoi(id);
        ++i;
    }
    ATNN_ASSERT_EQ(i, olen);
    return t;
}

struct Sample {
    std::string key;
    std::int64_t olen, ilen;
};

#include <cmath>
#include <limits>
auto make_batchset(json& data, size_t batch_size=32, size_t max_length_in=800, size_t max_length_out=150,
                   size_t max_num_batches=std::numeric_limits<size_t>::max()) {
    std::vector<Sample> keys;
    keys.reserve(data.size());
    for (auto d = data.begin(); d != data.end(); ++d) {
        Sample s;
        s.key = d.key();
        s.olen = get_int(d.value(), "olen");
        s.ilen = get_int(d.value(), "ilen");
        keys.push_back(s);
    } 
    // shorter first
    std::sort(keys.begin(), keys.end(),
              [](const auto& a, const auto& b) {
                  return a.olen < b.olen;  });
    std::vector<std::vector<Sample>> minibatch;
    auto start_id = 0;
    while (true) {
        const auto& start = keys[start_id];
        auto factor = std::max<int>(start.ilen / max_length_in, start.olen / max_length_out);
        auto b = std::max<int>(1, batch_size / (1 + factor));
        auto end_id = std::min<int>(keys.size(), start_id + b);
        std::vector<Sample> mb(keys.begin() + start_id, keys.begin() + end_id);
        minibatch.push_back(mb);
        if (end_id == keys.size() || minibatch.size() > max_num_batches) break;
        start_id = end_id;
    }
    return minibatch;
}

int main() {
    // testing float-vector io
    std::string train_dir = "wsj/dump/train_si284/deltafalse";
    std::string dev_dir = "wsj/dump/test_dev93/deltafalse";

    std::cout << "loading json" << std::endl;
    json train_json, dev_json;
    std::ifstream train_ifs(train_dir + "/data.json");
    train_ifs >> train_json;
    std::ifstream dev_ifs(dev_dir + "/data.json");
    dev_ifs >> dev_json;
    
    // see http://kaldi-asr.org/doc/classkaldi_1_1RandomAccessTableReader.html
    kaldi::RandomAccessBaseFloatMatrixReader train_reader("scp:" + train_dir + "/feats.scp");
    kaldi::RandomAccessBaseFloatMatrixReader dev_reader("scp:" + dev_dir + "/feats.scp");

    // validate data
    auto first_utt = dev_json["utts"].begin();
    auto test_id = first_utt.key();
    auto test_info = first_utt.value();
    std::cout << "validating data at uttid: " << test_id << std::endl;
    auto m = std::make_shared<kaldi::Matrix<float>>(dev_reader.Value(test_id));
    auto t = atnn::memory::make_tensor(m);
    // ilen (time) x idim (freq) (1082 x 83)
    auto ilen = std::stoi(test_info["ilen"].get<std::string>());
    auto idim = std::stoi(test_info["idim"].get<std::string>());
    ATNN_ASSERT_EQ(t.size(0), ilen);
    ATNN_ASSERT_EQ(t.size(1), idim);

    make_target(test_info);
    make_batchset(dev_json["utts"]);
}
