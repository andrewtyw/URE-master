from sentence_transformers import SentenceTransformer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_save_folder", type=str,default=None, help="model will download here")
args = parser.parse_args()

assert args.model_save_folder is not None

print("start to download huggingface models")
# helps to download the whole model, instead of the model cache
print("download microsoft/deberta-v2-xlarge-mnli...")
download_nli_model = SentenceTransformer("microsoft/deberta-v2-xlarge-mnli",cache_folder=args.model_save_folder)
print("download bert-base-uncased...")
download_bert_model = SentenceTransformer("bert-base-uncased",cache_folder=args.model_save_folder)