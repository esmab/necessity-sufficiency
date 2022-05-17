import numpy as np
import pandas as pd
import pickle
import argparse
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
import random
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
       # self.class_weights = torch.FloatTensor(class_weights)
        self.weighted_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).to(DEVICE)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = self.weighted_loss(logits, labels)
        if return_outputs:
            return loss, outputs
        else:
            return loss


def create_datasets(data_dir):
    # dataset = pickle.load(open(data_pickle, "rb"))
    # train_df = dataset['train'] #these are pd dataframes
    # valid_df = dataset['valid']
    # test_df = dataset['test']

    train_df = pd.read_csv(data_dir + "/train.csv")
    train_df = train_df.dropna()
    valid_df = pd.read_csv(data_dir + "/valid.csv")
    valid_df = valid_df.dropna()
    test_df = pd.read_csv(data_dir + "/test.csv")
    test_df = test_df.dropna()

    train_texts = train_df['text'].astype("string").tolist()
#  random.Random(42).shuffle(train_texts)
    valid_texts = valid_df['text'].astype("string").tolist()
    test_texts = test_df['text'].astype("string").tolist()

    train_labels = train_df['label'].astype("int").tolist()
    valid_labels = valid_df['label'].astype("int").tolist()
    test_labels = test_df['label'].astype("int").tolist()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # add special tokens for URLs, emojis and mentions (--> see pre-processing)
    special_tokens_dict = {'additional_special_tokens': ['[USER]', '[EMOJI]', '[URL]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")#.to(DEVICE)
    valid_encodings = tokenizer(valid_texts, padding=True, truncation=True, return_tensors="pt")#.to(DEVICE)
    test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

    train_dataset = HateDataset(train_encodings, train_labels)
    valid_dataset = HateDataset(valid_encodings, valid_labels)
    test_dataset = HateDataset(test_encodings, test_labels)

    return train_dataset, valid_dataset, test_dataset, len(tokenizer)


def calculate_class_weights(data_dir):
    dataset = pd.read_csv(data_dir + "/train.csv")
    train_labels = dataset.label.to_numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print("class weights are {}".format(class_weights))
    return class_weights


def train_model(train_dataset, valid_dataset, tok_len,  class_weights, output_dir, learning_rate, num_epochs, batch_size):

    training_args = TrainingArguments(
        save_steps=2500,
        output_dir=output_dir,  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        learning_rate=learning_rate,
        seed=123
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(DEVICE)
    model.resize_token_embeddings(tok_len)

    trainer = WeightedTrainer(
        model=model,
        class_weights=class_weights,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset
    # )
    try:
        trainer.train(resume_from_checkpoint=True)
        print("resuming from checkpoint...")
    except ValueError:
        print("No checkpoints found. training from scratch...")
        trainer.train()

    return trainer


if __name__ == '__main__':
    print("Starting training...")
    print("Training on {}".format(DEVICE))
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="A directory containing training validation and test sets to train the model on",
                        type=str)
    parser.add_argument("output_dir",
                        help="The directory to write output files to",
                        type=str)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("eval_only", action="store_true")
    parser.add_argument("--eval_on_test", action="store_true", help="Perform final evaluation on the test set")
    args = parser.parse_args()
    print("Training on the dataset: {}".format(args.dataset))
  #  pickle.dump(args, open(args.output_dir + "/training_arguments.pickle", "wb"))

    train_dataset, valid_dataset, test_dataset, tok_len = create_datasets(args.dataset)
    class_weights = calculate_class_weights(args.dataset)

    trainer = train_model(train_dataset,
                          valid_dataset,
                          tok_len,
                          class_weights,
                          args.output_dir,
                          args.learning_rate,
                          args.num_epochs,
                          args.batch_size)
    trainer.save_model(args.output_dir)

    print("Training done, evaluating...")
    valid_preds = np.argmax(trainer.predict(valid_dataset)[0], axis=1) #should be numpy ndarray
    valid_labels = np.array(valid_dataset.labels)

    cls_report_valid = classification_report(valid_labels, valid_preds, output_dict=True)
    pickle.dump(cls_report_valid, open(args.output_dir + "/cls_report_valid.pickle", "wb"))

    if args.eval_on_test:
        test_preds = np.argmax(trainer.predict(test_dataset)[0], axis=1)
        test_labels = np.array(test_dataset.labels)
        cls_report_test = classification_report(test_labels, test_preds, output_dict=True)
        pickle.dump(cls_report_test, open(args.output_dir + "/cls_report_test.pickle", "wb"))




