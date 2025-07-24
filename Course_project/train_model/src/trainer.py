import json
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from datasets import Dataset
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split


class KeywordExtractorTrainer:
    def __init__(self):
        """
        Initialize the KeywordExtractorTrainer class for training
        a BART model for keyword extraction.

        Attributes:
            device (str): The device to use for training
            ('cuda' or 'cpu').
            model_name (str): Name of the pretrained BART model.
            tokenizer (BartTokenizer): Tokenizer for the BART model.
            model (BartForConditionalGeneration): BART model for
            conditional generation.
            output_dir (Path): Directory to save trained models.
            logging_dir (Path): Directory to save training logs.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ilsilfverskiold/bart-keyword-extractor"
        self.tokenizer = None
        self.model = None
        self.output_dir = Path("./models/results")
        self.logging_dir = Path("./models/logs")

        nltk.download('punkt')
        nltk.download('stopwords')

    def load_data(self, data_path: Path) -> None:
        """
        Load and split the dataset into training and evaluation sets.

        Args:
            data_path (Path): Path to the JSON file containing
            the formatted data.
        """
        with open(data_path, "r") as f:
            data = json.load(f)

        train_data, temp_data = train_test_split(
            data, test_size=0.3, shuffle=True, random_state=42
            )
        eval_data, test_data = train_test_split(
            temp_data, test_size=0.5, shuffle=True, random_state=42
            )
    
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

    def initialize_model(self) -> None:
        """
        Initialize the tokenizer and model from
        the pretrained BART model.
        """
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name
            ).to(self.device)

    def preprocess_function(self, examples: dict) -> dict:
        """
        Preprocess the examples for model input.

        Args:
            examples (dict): Dictionary containing 'text' and 'keywords'.

        Returns:
            dict: Processed model inputs with tokenized text and labels.
        """
        inputs = [text for text in examples['text']]
        targets = [', '.join(keywords) for keywords in examples['keywords']]

        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True, padding='max_length'
        )

        labels = self.tokenizer(
            text_target=targets,
            max_length=128,
            truncation=True,
            padding='max_length'
        )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    def create_datasets(self) -> None:
        """
        Create training and evaluation datasets from the loaded data.
        """
        self.train_dataset = Dataset.from_dict({
            'text': [x['text'] for x in self.train_data],
            'keywords': [x['keywords'] for x in self.train_data]
        }).map(self.preprocess_function, batched=True)

        self.eval_dataset = Dataset.from_dict({
            'text': [x['text'] for x in self.eval_data],
            'keywords': [x['keywords'] for x in self.eval_data]
        }).map(self.preprocess_function, batched=True)

    def setup_training(self) -> None:
        """
        Set up training arguments and create the Seq2SeqTrainer.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            lr_scheduler_type="linear",
            warmup_steps=500,
            seed=42,
            optim="adamw_torch",
            logging_dir=str(self.logging_dir),
            logging_steps=100,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

    def compute_metrics(self, eval_pred: tuple) -> dict:
        """
        Compute evaluation metrics (precision, recall, F1)
        for keyword extraction.

        Args:
            eval_pred (tuple): Tuple containing predictions and labels.

        Returns:
            dict: Dictionary containing average precision, recall and
            F1 score.
        """
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
            )

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
            )

        pred_keywords = [set(word_tokenize(pred.lower()))
                         for pred in decoded_preds]
        true_keywords = [set(word_tokenize(label.lower()))
                         for label in decoded_labels]

        precisions, recalls, f1s = [], [], []
        for pred, true in zip(pred_keywords, true_keywords):
            if len(pred) == 0 or len(true) == 0:
                continue

            common = pred & true
            precision = len(common) / len(pred) if len(pred) > 0 else 0
            recall = len(common) / len(true) if len(true) > 0 else 0
            f1 = (2 * (precision * recall) / (precision + recall)
                  if (precision + recall) > 0 else 0)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s)
        }

    def train(self) -> None:
        """
        Train the model and evaluate on the evaluation set.
        """
        self.trainer.train()
        results = self.trainer.evaluate()

        report_path_train = self.output_dir.parent / "report_eval.json"
        with open(report_path_train, "w") as report_file:
            json.dump(results, report_file)

    def save_model(self) -> None:
        """
        Save the trained model and tokenizer.
        """
        model_dir = self.output_dir.parent / "final_model"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def evaluate_on_test(self) -> dict:
        """Evaluate the trained model on the held-out test dataset.

        This method creates a test dataset from the preloaded test data,
        preprocesses it using the same function as training data,
        and evaluates the model's performance on it.

        Returns:
            dict: A dictionary containing evaluation metrics such as
            precision, recall, and F1 score calculated on the test set.
            The exact metrics depend on the compute_metrics method
            implementation.
        """
        test_dataset = Dataset.from_dict({
            'text': [x['text'] for x in self.test_data],
            'keywords': [x['keywords'] for x in self.test_data]
        }).map(self.preprocess_function, batched=True)

        results = self.trainer.evaluate(test_dataset)

        report_path_test = self.output_dir.parent / "report_test.json"
        with open(report_path_test, "w") as report_file:
            json.dump(results, report_file)
        return results

    def extract_keywords(self, text: str, max_length: int = 128) -> list:
        """
        Extract keywords from a given text using the trained model.

        Args:
            text (str): Input text to extract keywords from.
            max_length (int): Maximum length for generated output.

        Returns:
            list: List of extracted keywords.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        output = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        ).to(self.device)

        keywords = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return keywords.split(', ')


if __name__ == "__main__":
    trainer = KeywordExtractorTrainer()
    trainer.load_data(Path("./data/formatted_data.json"))
    trainer.initialize_model()
    trainer.create_datasets()
    trainer.setup_training()
    trainer.train()
    trainer.save_model()

    test_results = trainer.evaluate_on_test()
    print("Test metrics:", test_results)

    # Example usage
    text = "Who is at risk for Pericarditis?"
    keywords = trainer.extract_keywords(text)
    print(f"Extracted keywords: {keywords}")
