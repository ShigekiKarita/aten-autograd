for setname in train valid test; do
    # wget https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/${setname}.txt
    wget https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.${setname}.txt -O ${setname}.txt
done

