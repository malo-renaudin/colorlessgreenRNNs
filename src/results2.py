import os
import sys
import pandas as pd
from syntactic_testsets.utils import load_vocab

def lstm_probs(output, gold, w2idx):
    data = []
    for scores, g in zip(output, gold):
        scores = scores.split()
        form, form_alt = g.split("\t")[6:8]
        prob_correct = float(scores[w2idx[form]])
        prob_wrong = float(scores[w2idx[form_alt]])
        data.append(prob_correct)
        data.append(prob_wrong)
    return data

lang = sys.argv[1]
path_repo = "/scratch2/mrenaudin/colorlessgreenRNNs/data"
path_test_data = os.path.join(path_repo, "agreement", lang, "generated")
path_output = os.path.join(path_repo, "agreement", lang, "generated.output_")
path_lm_data = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
path_results = "/scratch2/mrenaudin/colorlessgreenRNNs/results"
gold = open(path_test_data + ".gold").readlines()
sents = open(path_test_data + ".text").readlines()
data = pd.read_csv(path_test_data + ".tab", sep="\t")

vocab = load_vocab(os.path.join(path_lm_data, "vocab.txt"))

results = {"original": [], "generated": []}

# Loop over all model files
for file in os.listdir(os.path.dirname(path_output)):
    if file.startswith("generated.output_epoch"):
        model = file.replace("generated.output_", "")
        print(f"Processing model: {model}")
        
        if os.path.isfile(path_output + model):
            output = open(path_output + model).readlines()
            data[model] = lstm_probs(output, gold, vocab)

        if "freq" in data:
            models = [model, "freq"]
        else:
            models = [model]

        fields = ["pattern", "constr_id", "sent_id", "n_attr", "punct", "len_prefix", "len_context", "sent", "correct_number", "type"]
        wide_data = data[fields + ["class"] + models].pivot_table(columns=("class"), values=models, index=fields)

        for model in models:
            correct = wide_data.loc[:, (model, "correct")]
            wrong = wide_data.loc[:, (model, "wrong")]
            wide_data[(model, "acc")] = (correct > wrong) * 100

        t = wide_data.reset_index()
        a = t.groupby("type").agg({(m, "acc"): "mean" for m in models})
        print("Accuracy overall\n", a)

        # Compute accuracy by pattern, separating "original" and "generated"
        a_orig = (
            t[t.type == "original"]
            .groupby("pattern")
            .agg({(m, "acc"): "mean" for m in models})
            .rename(columns=lambda x: f"{x}_orig")
        )
        a_gen = (
            t[t.type == "generated"]
            .groupby("pattern")
            .agg({(m, "acc"): "mean" for m in models})
            .rename(columns=lambda x: f"{x}_gen")
        )

        a_final = pd.concat([a_orig, a_gen], axis=1).reset_index()
        a_final.insert(0, "model", model)

        results["original"].append(a_orig.assign(model=model).reset_index())
        results["generated"].append(a_gen.assign(model=model).reset_index())

# Save final CSVs
for result_type, dfs in results.items():
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_csv_path = os.path.join(path_results, f"{result_type}2_results.csv")
        final_df.to_csv(final_csv_path, sep="\t", index=False)
        print(f"Results saved to {final_csv_path}")