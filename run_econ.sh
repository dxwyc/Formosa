#python main.py --status decode \
#        --raw ./data/raw.text \
#        --savedset ./data/demo.dset \
#        --loadmodel ./data/demo.25.model \
#        --output ./data/demo.raw.out
#

python serve_econ.py --savedset ./data/econ_ner/demo.dset \
                --loadmodel ./data/econ_ner/demo.9.model

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/demo.test.char.tsv \
# 		--savedset ../data/onto4ner.cn/demo.dset \
# 		--loadmodel ../data/onto4ner.cn/demo.0.model \
# 		--output ../data/onto4ner.cn/demo.raw.out \
