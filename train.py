"""
train.py - Fine-tuning Script for Fake News Detection using DistilBERT / BERT
Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and saves model weights.
"""

import os
import re
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT/BERT on Fake News Dataset")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Pretrained Hugging Face model identifier (e.g., 'distilbert-base-uncased', 'bert-base-uncased')")
    parser.add_argument("--fake_csv", type=str, default="Fake.csv",
                        help="Path to Fake.csv from ISOT / Kaggle dataset")
    parser.add_argument("--true_csv", type=str, default="True.csv",
                        help="Path to True.csv from ISOT / Kaggle dataset")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Total samples to use (balanced). Use 0 to train on full dataset.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Per-device evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate for AdamW optimizer")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence token length")
    parser.add_argument("--output_dir", type=str, default="fake_news_model",
                        help="Directory to save the fine-tuned model and tokenizer")
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick synthetic demo training without requiring external CSV downloads")
    return parser.parse_args()


def clean_text(t: str) -> str:
    """Remove label-leaking Reuters headers, URLs, and extra whitespaces."""
    t = re.sub(r"^.{0,80}?\([rR]euters\)\s*[-—]?\s*", "", str(t))
    t = re.sub(r"\b[rR]euters\b", "", t)
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_dataset(fake_path: str, true_path: str, sample_size: int, demo_mode: bool = False):
    """Load, combine, clean, and stratify dataset."""
    import pandas as pd
    if demo_mode or not (os.path.exists(fake_path) and os.path.exists(true_path)):
        print("Note: CSV files not found or --demo mode selected. Creating synthetic training corpus for verification...")
        demo_fake = [
            "BREAKING: Secret documents prove alien technology is powering government agencies!",
            "SHOCKING: Miracle fruit eliminates all diseases overnight without medication!",
            "Mainstream media hides the truth about international banking conspiracy!",
            "Celebrity secretly arrested in underground tribunal, share before deletion!",
            "Unbelievable discovery reveals ancient pyramid power grid hidden from public!",
            "They do not want you to know this simple trick that banks hate!",
            "Leaked memo reveals massive election conspiracy by deep state officials!",
            "Miracle spice cures diabetes in 48 hours according to unverified blog!"
        ] * 25
        demo_true = [
            "The central bank maintained benchmark interest rates following monthly policy meeting.",
            "Federal regulators issued new guidelines on commercial aviation safety standards.",
            "Scientists published peer-reviewed climate data analyzing ocean surface temperatures.",
            "The foreign ministry issued a formal statement on diplomatic bilateral trade talks.",
            "Treasury officials announced quarterly economic growth statistics on Thursday.",
            "Health authorities recommended seasonal influenza vaccinations for high-risk demographics.",
            "The judicial committee concluded hearings regarding municipal infrastructure budgets.",
            "Astronomers detected atmospheric components on an exoplanet using space telescope."
        ] * 25

        df_fake = pd.DataFrame({"content": demo_fake, "label": 0})
        df_true = pd.DataFrame({"content": demo_true, "label": 1})
        df = pd.concat([df_fake, df_true], ignore_index=True)
    else:
        print(f"Loading '{fake_path}' and '{true_path}'...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df["label"] = 0
        true_df["label"] = 1

        df = pd.concat([fake_df, true_df], ignore_index=True)

        # Combine title and text
        title_col = df["title"].fillna("") if "title" in df.columns else ""
        text_col = df["text"].fillna("") if "text" in df.columns else ""
        df["content"] = title_col + ". " + text_col

    # Cleaning
    print("Preprocessing text and removing label-leaking artifacts...")
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.len() > 30]
    df = df.drop_duplicates(subset="content").reset_index(drop=True)

    if sample_size > 0 and len(df) > sample_size:
        half = sample_size // 2
        f_sub = df[df["label"] == 0].sample(min(half, len(df[df["label"] == 0])), random_state=42)
        t_sub = df[df["label"] == 1].sample(min(half, len(df[df["label"] == 1])), random_state=42)
        df = pd.concat([f_sub, t_sub], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final training corpus size: {len(df)} samples ({df['label'].value_counts().to_dict()})")
    return df[["content", "label"]]


def main():
    args = parse_args()
    print("=" * 60)
    print("Fake News Detection - DistilBERT / BERT Fine-Tuning Pipeline")
    print(f"Model: {args.model_name} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 60)

    try:
        import torch
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report
        )
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            DataCollatorWithPadding, TrainingArguments, Trainer
        )
    except ImportError as e:
        print(f"\nMissing required ML package: {e}")
        print("Please install requirements: pip install transformers datasets torch scikit-learn accelerate")
        return

    # 1. Load data
    df = load_dataset(args.fake_csv, args.true_csv, args.sample_size, args.demo)

    # 2. Stratified 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

    print(f"Train partition: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

    # 3. Tokenizer & Dataset creation
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(batch):
        return tokenizer(batch["content"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize_function, batched=True)
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize_function, batched=True)

    # 4. Model initialization
    print(f"Loading base classification model: {args.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1}
    )

    # 5. Compute metrics callback
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    training_kwargs = dict(
        output_dir="./training_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none"
    )

    try:
        training_args = TrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 6. Training
    print("\nStarting model fine-tuning...")
    trainer.train()

    # 7. Final Evaluation on Unseen Test Partition
    print("\n" + "=" * 60)
    print("Evaluating model on held-out test dataset...")
    print("=" * 60)
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["FAKE", "REAL"], output_dict=True)

    print(f"\nFinal Test Results:")
    print(f"  • Accuracy : {acc * 100:.2f}%")
    print(f"  • Precision: {prec * 100:.2f}%")
    print(f"  • Recall   : {rec * 100:.2f}%")
    print(f"  • F1-Score : {f1 * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # 8. Save Model & Tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_output = {
        "model_name": args.model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report
    }
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nModel and tokenizer successfully saved to: '{args.output_dir}'")
    print("You can now run the Streamlit app with:")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
