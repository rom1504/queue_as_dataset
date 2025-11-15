"""Apache Beam worker for parallel processing of queue items."""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional, List, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from .queue import QueueItem, SQLiteQueue
from .worker import WebPageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_from_queue(db_path: str, queue_name: str) -> Iterator[QueueItem]:
    """Generator function that reads items from the queue.

    Args:
        db_path: Path to SQLite database
        queue_name: Name of the queue

    Yields:
        QueueItem objects from the queue
    """
    queue = SQLiteQueue(db_path, queue_name)

    while True:
        # Pop item from queue
        item = queue.pop(visibility_timeout=600)  # 10 minute timeout

        if item is None:
            # No more items, we're done
            break

        yield item


class ProcessPageDoFn(beam.DoFn):
    """DoFn that processes web pages with optional threading."""

    def __init__(self, db_path: str, queue_name: str, output_dir: str = "data", num_threads: int = 1):
        self.db_path = db_path
        self.queue_name = queue_name
        self.output_dir = output_dir
        self.num_threads = num_threads
        self.processor = None
        self.queue = None
        self.executor = None

    def setup(self):
        """Initialize processor and queue (called once per worker)."""
        logger.info(f"Setting up processor in worker {id(self)} with {self.num_threads} threads")
        self.processor = WebPageProcessor(output_dir=self.output_dir)
        self.queue = SQLiteQueue(self.db_path, self.queue_name)
        if self.num_threads > 1:
            self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        else:
            self.executor = None

    def teardown(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

    def process(self, item: QueueItem):
        """Process a single queue item.

        Args:
            item: QueueItem to process

        Yields:
            Tuple of (item_id, success, url, num_chunks)
        """
        if self.executor is None:
            # Single-threaded: process immediately
            yield from self._process_item(item)
        else:
            # Multi-threaded: submit to executor
            # Note: This will still be called sequentially by Beam,
            # but we can process multiple items concurrently per worker
            future = self.executor.submit(self._process_item_sync, item)
            result = future.result()  # Wait for completion
            yield result

    def _process_item(self, item: QueueItem):
        """Process item and yield result."""
        result = self._process_item_sync(item)
        yield result

    def _process_item_sync(self, item: QueueItem) -> Tuple[int, bool, str, int]:
        """Synchronously process a single queue item.

        Args:
            item: QueueItem to process

        Returns:
            Tuple of (item_id, success, url, num_chunks)
        """
        url = item.payload.get("page_url")
        if not url:
            logger.error(f"Item {item.id} has no page_url")
            self.queue.fail(item.id, "Missing page_url in payload")
            return (item.id, False, "", 0)

        try:
            start_time = time.time()

            # Process the page
            processed = self.processor.process_page(url)

            if processed is None:
                self.queue.fail(item.id, "Failed to process page")
                return (item.id, False, url, 0)

            # Save to disk
            filepath = self.processor.save_processed_page(processed, item.id)

            # Update metadata
            item.metadata["output_file"] = filepath
            item.metadata["num_chunks"] = len(processed.chunks)
            item.metadata["completed_at"] = time.time()

            # Acknowledge success
            self.queue.ack(item.id, metadata=item.metadata)

            elapsed = time.time() - start_time
            logger.info(
                f"Worker {id(self)}: Processed item {item.id}: {url} "
                f"({len(processed.chunks)} chunks) in {elapsed:.2f}s"
            )

            return (item.id, True, url, len(processed.chunks))

        except Exception as e:
            logger.error(f"Worker {id(self)}: Error processing item {item.id}: {e}", exc_info=True)
            self.queue.fail(item.id, str(e))
            return (item.id, False, url, 0)


class ProcessBatchFn(beam.DoFn):
    """DoFn that processes batches of items with threading."""

    def __init__(self, db_path: str, queue_name: str, output_dir: str = "data", num_threads: int = 1):
        self.db_path = db_path
        self.queue_name = queue_name
        self.output_dir = output_dir
        self.num_threads = num_threads
        self.processor = None
        self.queue = None

    def setup(self):
        """Initialize processor and queue (called once per worker)."""
        logger.info(f"Setting up processor in worker {id(self)} with {self.num_threads} threads")
        self.processor = WebPageProcessor(output_dir=self.output_dir)
        self.queue = SQLiteQueue(self.db_path, self.queue_name)

    def process(self, batch: List[QueueItem]):
        """Process a batch of queue items with threading.

        Args:
            batch: List of QueueItems to process

        Yields:
            Tuples of (item_id, success, url, num_chunks)
        """
        if self.num_threads == 1 or len(batch) == 1:
            # Single-threaded: process sequentially
            for item in batch:
                yield self._process_item_sync(item)
        else:
            # Multi-threaded: process batch concurrently
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self._process_item_sync, item) for item in batch]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        yield result
                    except Exception as e:
                        logger.error(f"Error in threaded processing: {e}", exc_info=True)

    def _process_item_sync(self, item: QueueItem) -> Tuple[int, bool, str, int]:
        """Synchronously process a single queue item.

        Args:
            item: QueueItem to process

        Returns:
            Tuple of (item_id, success, url, num_chunks)
        """
        url = item.payload.get("page_url")
        if not url:
            logger.error(f"Item {item.id} has no page_url")
            self.queue.fail(item.id, "Missing page_url in payload")
            return (item.id, False, "", 0)

        try:
            start_time = time.time()

            # Process the page
            processed = self.processor.process_page(url)

            if processed is None:
                self.queue.fail(item.id, "Failed to process page")
                return (item.id, False, url, 0)

            # Save to disk
            filepath = self.processor.save_processed_page(processed, item.id)

            # Update metadata
            item.metadata["output_file"] = filepath
            item.metadata["num_chunks"] = len(processed.chunks)
            item.metadata["completed_at"] = time.time()

            # Acknowledge success
            self.queue.ack(item.id, metadata=item.metadata)

            elapsed = time.time() - start_time
            logger.info(
                f"Worker {id(self)}: Processed item {item.id}: {url} "
                f"({len(processed.chunks)} chunks) in {elapsed:.2f}s"
            )

            return (item.id, True, url, len(processed.chunks))

        except Exception as e:
            logger.error(f"Worker {id(self)}: Error processing item {item.id}: {e}", exc_info=True)
            self.queue.fail(item.id, str(e))
            return (item.id, False, url, 0)


def run_beam_pipeline(
    db_path: str,
    queue_name: str,
    output_dir: str,
    num_workers: int = 4,
    num_threads: int = 1,
    batch_size: int = 10,
    runner: str = "DirectRunner",
):
    """Run the Beam pipeline for parallel processing.

    Args:
        db_path: Path to SQLite database
        queue_name: Name of the queue
        output_dir: Output directory for processed files
        num_workers: Number of parallel workers (processes)
        num_threads: Number of threads per worker
        batch_size: Number of items per batch for threading
        runner: Beam runner to use (DirectRunner, DataflowRunner, etc.)
    """
    # Set up pipeline options
    pipeline_options = PipelineOptions(
        runner=runner,
        direct_num_workers=num_workers,
        direct_running_mode="multi_threading",
    )
    pipeline_options.view_as(SetupOptions).save_main_session = True

    logger.info(f"Starting Beam pipeline with {num_workers} workers, {num_threads} threads per worker using {runner}")
    logger.info(f"Processing queue: {queue_name} from {db_path}")
    logger.info(f"Batch size: {batch_size} items per batch")

    # Create and run pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read from queue - collect all items upfront
        queue_items = list(read_from_queue(db_path, queue_name))
        logger.info(f"Found {len(queue_items)} items to process")

        # Batch all items upfront
        batched_items = []
        for i in range(0, len(queue_items), batch_size):
            batched_items.append(queue_items[i:i+batch_size])

        logger.info(f"Created {len(batched_items)} batches of ~{batch_size} items each")

        # Create PCollection from batches
        batches = pipeline | "Create Batches" >> beam.Create(batched_items)

        # Process batches in parallel with threading
        results = batches | "Process Pages" >> beam.ParDo(
            ProcessBatchFn(db_path, queue_name, output_dir, num_threads)
        )

        # Collect statistics
        def combine_stats(stats_list):
            """Combine statistics from all processed items."""
            total_success = sum(s["success"] for s in stats_list)
            total_failed = sum(s["failed"] for s in stats_list)
            total_chunks = sum(s["chunks"] for s in stats_list)
            return {
                "success": total_success,
                "failed": total_failed,
                "chunks": total_chunks,
            }

        stats = (
            results
            | "Extract Stats" >> beam.Map(lambda x: {
                "success": 1 if x[1] else 0,
                "failed": 0 if x[1] else 1,
                "chunks": x[3],
            })
            | "Combine Stats" >> beam.CombineGlobally(combine_stats)
        )

        # Log final stats
        stats | "Log Stats" >> beam.Map(
            lambda s: logger.info(
                f"Pipeline complete: {s['success']} succeeded, "
                f"{s['failed']} failed, {s['chunks']} total chunks"
            )
        )


def main():
    """Main entry point for Beam worker."""
    parser = argparse.ArgumentParser(
        description="Apache Beam worker for parallel queue processing"
    )
    parser.add_argument(
        "--db",
        default="queue.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--queue",
        default="page_queue",
        help="Queue name"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (processes)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads per worker (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of items per batch for threading (default: 10)"
    )
    parser.add_argument(
        "--runner",
        default="DirectRunner",
        choices=["DirectRunner", "DataflowRunner", "FlinkRunner"],
        help="Beam runner to use"
    )

    args = parser.parse_args()

    start_time = time.time()

    run_beam_pipeline(
        db_path=args.db,
        queue_name=args.queue,
        output_dir=args.output_dir,
        num_workers=args.workers,
        num_threads=args.threads,
        batch_size=args.batch_size,
        runner=args.runner,
    )

    elapsed = time.time() - start_time
    logger.info(f"Total pipeline execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
