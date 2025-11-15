"""Worker for training a transformer model to predict chunk counts from page JSON."""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import wandb

from .queue import QueueItem, SQLiteQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PageJSONDataset(Dataset):
    """Dataset of page JSON strings and their chunk counts."""

    def __init__(self, json_texts: List[str], chunk_counts: List[int], tokenizer, max_length: int = 512):
        self.json_texts = json_texts
        self.chunk_counts = chunk_counts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Bin chunk counts into classes for classification
        # Bins: 0-20, 21-50, 51-100, 101-200, 200+
        self.num_classes = 5

    def _bin_chunk_count(self, count: int) -> int:
        """Convert chunk count to class bin."""
        if count <= 20:
            return 0
        elif count <= 50:
            return 1
        elif count <= 100:
            return 2
        elif count <= 200:
            return 3
        else:
            return 4

    def __len__(self):
        return len(self.json_texts)

    def __getitem__(self, idx):
        text = self.json_texts[idx]
        label = self._bin_chunk_count(self.chunk_counts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class MLTrainer:
    """Trains a transformer model to predict chunk counts from page JSON."""

    def __init__(
        self,
        input_dir: str = "data",
        model_dir: str = "models/chunk_predictor",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        batch_size: int = 8,
        num_train_epochs: int = 3,
        max_steps: int = -1
    ):
        self.input_dir = Path(input_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.num_classes = 5  # Chunk count bins

        # Check if trained model exists
        model_exists = (self.model_dir / "config.json").exists()

        # Initialize tokenizer
        if model_exists:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize or load model
        if model_exists:
            logger.info(f"Loading existing model from {self.model_dir}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                num_labels=self.num_classes
            )
        else:
            logger.info(f"Initializing new model: {model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes
            )

    def load_training_data(self, processed_files: List[str]) -> tuple:
        """Load JSON texts and chunk counts from processed files.

        Args:
            processed_files: List of paths to processed JSON files

        Returns:
            Tuple of (json_texts, chunk_counts)
        """
        json_texts = []
        chunk_counts = []

        for filepath in processed_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Use the entire JSON as text input (serialized)
                json_text = json.dumps(data, separators=(',', ':'))  # Compact JSON

                # Get chunk count as label
                chunks = data.get('chunks', [])
                chunk_count = len(chunks)

                json_texts.append(json_text)
                chunk_counts.append(chunk_count)

            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                continue

        return json_texts, chunk_counts

    def train(self, json_texts: List[str], chunk_counts: List[int]) -> Dict:
        """Train the model on the provided data.

        Args:
            json_texts: List of JSON strings
            chunk_counts: List of chunk counts

        Returns:
            Training metrics dictionary
        """
        if len(json_texts) < 10:
            logger.warning(f"Only {len(json_texts)} examples, skipping training")
            return {"status": "skipped", "reason": "insufficient_data", "num_examples": len(json_texts)}

        logger.info(f"Training on {len(json_texts)} examples")

        # Initialize wandb
        wandb.init(
            project="chunk-count-predictor",
            name=f"chunk_predictor_{time.time():.0f}",
            config={
                "model_name": self.model_name,
                "num_examples": len(json_texts),
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "num_train_epochs": self.num_train_epochs,
                "max_steps": self.max_steps,
            }
        )

        # Create dataset
        dataset = PageJSONDataset(json_texts, chunk_counts, self.tokenizer, self.max_length)

        # Split: use 10 samples for eval, rest for training
        eval_size = min(10, len(dataset) // 10)  # 10 samples or 10% if dataset is small
        train_size = len(dataset) - eval_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size]
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "checkpoints"),
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=10 if self.max_steps > 0 else 100,
            weight_decay=0.01,
            logging_dir=str(self.model_dir / "logs"),
            logging_steps=5,
            eval_strategy="no",  # Only evaluate at the end
            save_strategy="no",  # Don't save checkpoints during training
            load_best_model_at_end=False,
            report_to="wandb",
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        train_result = trainer.train()

        # Save model
        trainer.save_model(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))

        # Evaluate
        eval_result = trainer.evaluate()

        metrics = {
            "status": "completed",
            "num_examples": len(json_texts),
            "train_size": train_size,
            "eval_size": eval_size,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result.get("eval_loss"),
            "epochs": self.num_train_epochs,
            "model_dir": str(self.model_dir),
        }

        logger.info(f"Training complete: train_loss={metrics['train_loss']:.4f}, eval_loss={metrics['eval_loss']:.4f}")

        # Finish wandb run
        wandb.finish()

        return metrics

    def predict(self, json_text: str) -> Dict:
        """Run inference on a single JSON text.

        Args:
            json_text: JSON string to classify

        Returns:
            Dictionary with prediction results
        """
        self.model.eval()

        # Tokenize
        encoding = self.tokenizer(
            json_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

        # Convert class to chunk count range
        chunk_ranges = {
            0: "0-20",
            1: "21-50",
            2: "51-100",
            3: "101-200",
            4: "200+"
        }

        return {
            "predicted_class": predicted_class,
            "predicted_range": chunk_ranges[predicted_class],
            "confidence": confidence,
            "probabilities": probs[0].tolist()
        }


def process_completed_items(
    source_queue_name: str,
    target_queue_name: str,
    db_path: str = "queue.db",
    input_dir: str = "data",
    model_dir: str = "models/chunk_predictor",
    batch_size: int = 8,
    min_examples: int = 50,
    max_steps: int = -1,
):
    """Process completed items from source queue and train model.

    Args:
        source_queue_name: Queue to read completed items from
        target_queue_name: Queue to push training results to
        db_path: Path to SQLite database
        input_dir: Directory with processed files
        model_dir: Directory to save model
        batch_size: Training batch size
        min_examples: Minimum examples needed before training
    """
    source_queue = SQLiteQueue(db_path, source_queue_name)
    target_queue = SQLiteQueue(db_path, target_queue_name)
    trainer = MLTrainer(input_dir=input_dir, model_dir=model_dir, batch_size=batch_size, max_steps=max_steps)

    logger.info(f"Starting ML training worker: {source_queue_name} → {target_queue_name}")
    logger.info(f"Model dir: {model_dir}, Min examples: {min_examples}")

    # Get completed items from source queue
    completed_items = source_queue.get_items(status='completed', limit=1000)
    logger.info(f"Found {len(completed_items)} completed items to use for training")

    if len(completed_items) < min_examples:
        logger.warning(f"Only {len(completed_items)} examples available, need at least {min_examples}")
        return

    # Collect processed files
    processed_files = []
    for item in completed_items:
        output_file = item['metadata'].get('output_file')
        if output_file and os.path.exists(output_file):
            processed_files.append(output_file)

    logger.info(f"Collected {len(processed_files)} valid processed files")

    if len(processed_files) < min_examples:
        logger.warning(f"Only {len(processed_files)} valid files, need at least {min_examples}")
        return

    # Load training data
    json_texts, chunk_counts = trainer.load_training_data(processed_files)
    logger.info(f"Loaded {len(json_texts)} training examples")

    # Train model
    metrics = trainer.train(json_texts, chunk_counts)

    # Create queue item with training results
    queue_item = QueueItem(
        payload={
            'model_type': 'chunk_predictor',
            'num_examples': len(json_texts),
            'metrics': metrics,
        },
        metadata={
            'source': 'ml_trainer',
            'training_metrics': metrics,
            'completed_at': time.time(),
        },
    )

    # Push to target queue and mark as completed
    target_item_id = target_queue.push(queue_item)
    target_queue.ack(target_item_id, metadata=queue_item.metadata)

    logger.info(f"Training complete and results saved to {target_queue_name}")


def run_inference_on_items(
    source_queue_name: str,
    target_queue_name: str,
    db_path: str = "queue.db",
    input_dir: str = "data",
    model_dir: str = "models/chunk_predictor",
    batch_size: int = 100,
):
    """Run inference on completed items from source queue.

    Args:
        source_queue_name: Queue to read completed items from
        target_queue_name: Queue to push predictions to
        db_path: Path to SQLite database
        input_dir: Directory with processed files
        model_dir: Directory with trained model
        batch_size: Number of items to process
    """
    source_queue = SQLiteQueue(db_path, source_queue_name)
    target_queue = SQLiteQueue(db_path, target_queue_name)

    # Check if model exists
    model_path = Path(model_dir)
    if not (model_path / "config.json").exists():
        logger.error(f"No trained model found at {model_dir}. Please train a model first.")
        return

    # Load model
    predictor = MLTrainer(input_dir=input_dir, model_dir=model_dir)
    logger.info(f"Loaded model from {model_dir}")

    logger.info(f"Starting ML inference worker: {source_queue_name} → {target_queue_name}")

    # Get completed items from source queue
    completed_items = source_queue.get_items(status='completed', limit=batch_size)
    logger.info(f"Found {len(completed_items)} completed items for inference")

    if not completed_items:
        logger.info("No completed items to process")
        return

    # Process each item
    predictions_made = 0
    for item in completed_items:
        try:
            # Load the processed JSON file
            output_file = item['metadata'].get('output_file')
            if not output_file or not os.path.exists(output_file):
                logger.warning(f"Item {item['id']}: No output file found, skipping")
                continue

            # Read JSON
            with open(output_file, 'r') as f:
                data = json.load(f)

            # Serialize JSON for prediction
            json_text = json.dumps(data, separators=(',', ':'))

            # Run inference
            prediction = predictor.predict(json_text)

            # Get actual chunk count for comparison
            actual_chunks = len(data.get('chunks', []))

            # Create queue item with prediction
            queue_item = QueueItem(
                payload={
                    'page_url': item['payload'].get('page_url'),
                    'prediction': prediction,
                    'actual_chunks': actual_chunks,
                    'source_item_id': item['id'],
                },
                metadata={
                    'source': 'ml_inference',
                    'model_dir': model_dir,
                    'predicted_range': prediction['predicted_range'],
                    'confidence': prediction['confidence'],
                    'actual_chunks': actual_chunks,
                    'completed_at': time.time(),
                },
            )

            # Push to target queue and mark as completed
            target_item_id = target_queue.push(queue_item)
            target_queue.ack(target_item_id, metadata=queue_item.metadata)

            predictions_made += 1
            logger.info(
                f"Item {item['id']}: Predicted {prediction['predicted_range']} chunks "
                f"(confidence: {prediction['confidence']:.3f}, actual: {actual_chunks})"
            )

        except Exception as e:
            logger.error(f"Error processing item {item['id']}: {e}", exc_info=True)
            continue

    logger.info(f"Inference complete: {predictions_made} predictions saved to {target_queue_name}")


def run_ml_worker(
    source_queue_name: str = "page_queue",
    target_queue_name: str = "ml_queue",
    db_path: str = "queue.db",
    input_dir: str = "data",
    model_dir: str = "models/chunk_predictor",
    interval: int = 300,  # Train every 5 minutes
    batch_size: int = 8,
    min_examples: int = 50,
    max_steps: int = -1,
):
    """Run ML worker that continuously trains on completed items.

    Args:
        source_queue_name: Queue to read completed items from
        target_queue_name: Queue to push training results to
        db_path: Path to SQLite database
        input_dir: Directory with processed files
        model_dir: Directory to save model
        interval: Seconds to wait between training runs
        batch_size: Training batch size
        min_examples: Minimum examples needed before training
    """
    logger.info(f"Starting continuous ML training worker (training every {interval}s)")

    while True:
        try:
            process_completed_items(
                source_queue_name=source_queue_name,
                target_queue_name=target_queue_name,
                db_path=db_path,
                input_dir=input_dir,
                model_dir=model_dir,
                batch_size=batch_size,
                min_examples=min_examples,
                max_steps=max_steps,
            )
        except Exception as e:
            logger.error(f"Error in ML worker loop: {e}", exc_info=True)

        logger.info(f"Waiting {interval} seconds before next training run...")
        time.sleep(interval)


def main():
    """Main entry point for ML worker."""
    parser = argparse.ArgumentParser(
        description="ML worker for training chunk count predictor"
    )
    parser.add_argument(
        "--source-queue",
        default="page_queue",
        help="Queue to read completed items from"
    )
    parser.add_argument(
        "--target-queue",
        default="ml_queue",
        help="Queue to push training results to"
    )
    parser.add_argument(
        "--db",
        default="queue.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory with processed files"
    )
    parser.add_argument(
        "--model-dir",
        default="models/chunk_predictor",
        help="Directory to save model"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Train once and exit (don't run continuously)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds to wait between training runs (for continuous mode)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=50,
        help="Minimum examples needed before training"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (default: -1 = use num_train_epochs)"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference mode instead of training"
    )

    args = parser.parse_args()

    if args.inference:
        # Run inference on completed items
        run_inference_on_items(
            source_queue_name=args.source_queue,
            target_queue_name=args.target_queue,
            db_path=args.db,
            input_dir=args.input_dir,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
        )
    elif args.once:
        process_completed_items(
            source_queue_name=args.source_queue,
            target_queue_name=args.target_queue,
            db_path=args.db,
            input_dir=args.input_dir,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            min_examples=args.min_examples,
            max_steps=args.max_steps,
        )
    else:
        run_ml_worker(
            source_queue_name=args.source_queue,
            target_queue_name=args.target_queue,
            db_path=args.db,
            input_dir=args.input_dir,
            model_dir=args.model_dir,
            interval=args.interval,
            batch_size=args.batch_size,
            min_examples=args.min_examples,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
