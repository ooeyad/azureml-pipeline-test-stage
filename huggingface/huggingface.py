import pandas as pd
from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import os
from pathlib import Path
import argparse

# dataset1 = pd.read_csv(f"{Path(__file__).parent}/cairs_processed.csv")

parser = argparse.ArgumentParser("huggingface")
parser.add_argument("--prepped_data", type=str, help="Path to raw data")
parser.add_argument("--status_output", type=str, help="Path of prepped data")
args = parser.parse_args()

filename = os.listdir(args.prepped_data)
dataset1 = pd.read_csv((Path(args.prepped_data) / filename[0]))

dataset1.to_csv((Path(args.status_output) / "status_output.csv"), index = False)

t1=dataset1.shape[0]
t_rain = int(t1 * 80 / 100)
t_est = int(t1 * 15 / 100)
v_alid = int(t1 * 5 / 100)
train1 = dataset1.sample(n=t_rain)
test1 = dataset1.sample(n=t_est)
validation1 = dataset1.sample(n=v_alid)
train = Dataset.from_dict(train1)
test=Dataset.from_dict(test1)
validation=Dataset.from_dict(validation1)
dataset=datasets.DatasetDict({"train": train, "test": test, "validation": validation})

def preprocess_data(examples):
    # take a batch of texts
    text = examples["CONCATENATED_TEXT"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

# @command_component(
#     name="send_to_hugging_face",
#     version="1",
#     display_name="Data Preparation",
#     description="preparing etc..",
#     environment="azureml:TestNew@latest"
# )

labels = [label for label in dataset['train'].features.keys() if label not in ['CONCATENATED_TEXT']]
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained("yashveer11/final_model_category")

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

example = encoded_dataset['train'][0]

tokenizer.decode(example['input_ids'])

# [id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]

encoded_dataset.set_format("torch")

access_token = "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"
login(access_token)

model = AutoModelForSequenceClassification.from_pretrained("yashveer11/final_model_category",
                                                            problem_type="multi_label_classification",
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)

batch_size = 8
metric_name = "f1"
from huggingface_hub import delete_repo
from huggingface_hub import HfApi
hf_api = HfApi()

model_exists = "yashveer11/testing_class" in [model.modelId for model in hf_api.list_models()]
my_model = ''
if model_exists:
    delete_repo(repo_id="yashveer11/testing_class")
args = TrainingArguments(
    f"testing_class",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=True,
    hub_model_id="yashveer11/testing_class"
)

encoded_dataset['train'][0]['labels'].type()
encoded_dataset['train']['input_ids'][0]

outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0),
                labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.push_to_hub("End of training")
