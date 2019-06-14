#python main.py --status train \
#		--train ./data/msra_ner/train.char.tsv \
#		--dev ./data/msra_ner/dev.char.tsv \
#		--test ./data/msra_ner/test.char.tsv \
#		--savemodel ./data/msra_ner/demo \

python main.py --status train \
		--train ./data/demo.train.char \
		--dev ./data/demo.dev.char \
		--test ./data/demo.test.char \
		--savemodel ./data/demo \

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/demo.test.char.tsv \
# 		--savedset ../data/onto4ner.cn/demo.dset \
# 		--loadmodel ../data/onto4ner.cn/demo.0.model \
# 		--output ../data/onto4ner.cn/demo.raw.out \
