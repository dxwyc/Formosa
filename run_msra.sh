python main.py --status inference \
        --raw ./data/raw2.text \
        --savedset ./data/msra_ner/msra.dset \
        --loadmodel ./data/msra_ner/msra.16.model \
        --output ./data/demo.raw.out

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/demo.test.char.tsv \
# 		--savedset ../data/onto4ner.cn/demo.dset \
# 		--loadmodel ../data/onto4ner.cn/demo.0.model \
# 		--output ../data/onto4ner.cn/demo.raw.out \
