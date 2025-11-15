# Queue as Dataset: A Pattern for Building Robust Data Processing Pipelines

## The Problem

When building machine learning systems or data processing pipelines, we often face a common set of challenges:

- **Progress tracking**: How do we know how much work is done and how much remains?
- **Fault tolerance**: What happens when a processing step fails for some items?
- **Visibility**: Can we inspect what's happening at each stage of the pipeline?
- **Resumability**: If we stop the process, can we resume without reprocessing everything?
- **Scalability**: How do we parallelize the work across multiple workers?
- **Data lineage**: Can we trace a data item through all processing stages?

Traditional approaches often involve writing custom scripts with manual checkpointing, logging to files, and ad-hoc retry logic. This leads to fragile, hard-to-maintain code.

## The Queue-as-Dataset Pattern

The core insight is simple: **treat your dataset as a persistent queue**. Instead of processing data in a one-shot script, you:

1. Push items into a queue (stored in a database like SQLite)
2. Workers pull items, process them, and mark them as complete
3. Each processing stage has its own queue
4. The queue itself becomes your dataset

This pattern transforms data processing from a script you run into a **system you operate**.

## Key Benefits

### 1. Built-in Fault Tolerance

When a worker crashes or an API call fails, the item remains in the queue. No data is lost. Workers can retry failed items with exponential backoff:

```python
item = queue.pop()  # Get next item
try:
    result = process(item)
    queue.mark_completed(item, result)
except Exception as e:
    queue.mark_failed(item, error=str(e))
```

In this project, if web scraping fails for a URL, it's marked as failed but remains in the database. You can later retry just the failed items without reprocessing everything.

### 2. Real-Time Visibility

Since the queue is a database, you can query it at any time:
- How many items are pending, processing, completed, or failed?
- What's the current processing speed?
- What errors are occurring?
- Can I see the details of a specific item?

The included web UI shows all of this in real-time. You can click any item to see its data at every processing stage.

### 3. Multi-Stage Pipelines

Complex processing becomes a series of simple stages. In this project:

1. **Page Queue**: Scrape web pages → Extract interleaved text/images
2. **Stats Queue**: Compute statistics on extracted content
3. **ML Queue**: Train model to predict chunk counts → Run inference

Each stage is independent. You can:
- Develop and test stages separately
- Scale each stage independently
- Add new stages without changing existing ones
- Reprocess specific stages if you improve the logic

### 4. Progressive Dataset Building

Your dataset builds up incrementally. You don't need all data before starting:

```python
# Add URLs as you discover them
queue.push({"page_url": "https://example.com"})

# Start processing immediately
worker.start()  # Processes whatever is available

# Add more URLs later - they'll be processed automatically
queue.push({"page_url": "https://another.com"})
```

This is perfect for:
- Web scraping (find more URLs as you go)
- ML training (train on early data while collecting more)
- Streaming data sources

### 5. Resumable Processing

Stop and restart anytime without losing progress:

```bash
# Start processing
python -m queue_processor.beam_worker --workers 4

# Ctrl+C to stop

# Hours later, resume exactly where you left off
python -m queue_processor.beam_worker --workers 4
```

Only pending items are processed. Completed items are skipped. This is invaluable for long-running jobs.

### 6. Data Lineage and Debugging

Every item knows:
- When it was created
- When it was processed
- What the result was
- What errors occurred (if any)
- Metadata about the processing

For ML workflows, you can trace a prediction back to:
- The original web page URL
- The extracted content (text/images)
- The computed statistics
- The model's prediction

This makes debugging much easier. When a model makes a bad prediction, you can inspect the entire processing chain.

### 7. Built-in Dataset Versioning

Want to reprocess with improved logic? Keep the old data:

```python
# Original processing
stats = compute_stats_v1(content)

# Later: Improved version
stats = compute_stats_v2(content)
```

You can:
- Compare old vs new results
- Gradually migrate to the new version
- Roll back if needed

### 8. Parallel Processing with Coordination

Multiple workers can process the queue simultaneously:

```python
# Worker 1
worker = Worker(queue_name="page_queue", worker_id=1)

# Worker 2 (different machine, same database)
worker = Worker(queue_name="page_queue", worker_id=2)
```

The queue handles coordination automatically:
- No duplicate processing (atomic pop operations)
- No race conditions
- Graceful handling of worker failures

In this project, the beam worker uses 4 workers with 4 threads each for parallel web scraping.

### 9. ML Training as a Processing Stage

The ML worker demonstrates a powerful pattern:

1. Wait until enough examples accumulate in stats_queue
2. Train a model to predict chunk counts from statistics
3. Push predictions to ml_queue
4. The queue tracks prediction accuracy

This turns ML training from a separate offline process into just another stage in your pipeline:

```python
# ml_worker.py - Train when enough data is ready
if stats_queue.count() >= min_examples:
    dataset = stats_queue.get_completed_items()
    model = train(dataset)

    # Run inference on new items
    for item in page_queue.get_items():
        prediction = model.predict(item)
        ml_queue.push({"page_url": item.url, "prediction": prediction})
```

### 10. Cost-Effective Experimentation

APIs are expensive. With the queue pattern:

```python
# First pass: Scrape and store raw data
for url in urls:
    page_queue.push({"page_url": url})

# Raw data is now cached in queue

# Experiment with different processing - NO additional API calls
def new_stats_extractor(content):
    # Try different statistics
    pass

# Reprocess from cached data
for item in page_queue.get_completed():
    new_stats = new_stats_extractor(item.content)
```

In this project, web pages are scraped once. You can experiment with different statistics or ML models without re-scraping.

## Implementation: SQLite as a Queue

The key is using a database table as a queue:

```sql
CREATE TABLE page_queue (
    id INTEGER PRIMARY KEY,
    status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    payload TEXT,  -- JSON data
    metadata TEXT, -- Processing results, errors, etc.
    created_at REAL,
    updated_at REAL
)
```

Core operations:

```python
# Push: Add item
def push(self, item):
    cursor.execute(
        f"INSERT INTO {self.queue_name} (status, payload, metadata) VALUES (?, ?, ?)",
        ('pending', json.dumps(item.payload), json.dumps(item.metadata))
    )

# Pop: Get next item atomically
def pop(self):
    # Atomic: Mark as processing and return in one transaction
    cursor.execute(f"""
        UPDATE {self.queue_name}
        SET status = 'processing', updated_at = ?
        WHERE id = (
            SELECT id FROM {self.queue_name}
            WHERE status = 'pending'
            LIMIT 1
        )
        RETURNING *
    """, (time.time(),))
```

SQLite is perfect for this because:
- Single file - easy to backup, share, deploy
- ACID transactions - no corruption
- Concurrent readers, single writer - good enough for most use cases
- No server setup required
- Can handle millions of items

For higher throughput, you could swap to PostgreSQL or Redis without changing the pattern.

## Real-World Example: Wikipedia Processing Pipeline

This project demonstrates the pattern with Wikipedia articles:

**Stage 1: Page Queue (Scraping)**
- Input: Wikipedia URLs
- Worker: beam_worker with 16 parallel connections
- Output: Interleaved text/image content as JSON
- Storage: data/processed_*.json files
- Metadata: chunk_count, processing_time

**Stage 2: Stats Queue (Analysis)**
- Input: Completed items from page_queue
- Worker: stats_worker polling every 10s
- Output: Statistics (total_chunks, text_chunks, image_chunks, total_words, avg_words_per_chunk)
- Metadata: Computed stats stored in queue

**Stage 3: ML Queue (Prediction)**
- Input: Items from stats_queue (for training), page_queue (for inference)
- Worker: ml_worker with DistilBERT
- Training: Predict chunk_count from content
- Output: Predictions with confidence scores
- Metadata: MAE, accuracy within thresholds

The web UI ties it together: click any item to see its journey through all three stages.

## When to Use This Pattern

**Great for:**
- Web scraping and data collection
- ETL pipelines with multiple transformation stages
- ML training where data arrives incrementally
- Long-running batch jobs that need to be resumable
- Processes that interact with rate-limited APIs
- Systems where you need audit trails and data lineage

## Conclusion

The queue-as-dataset pattern transforms data processing from fragile scripts into robust systems. By treating your dataset as a persistent queue, you get fault tolerance, visibility, resumability, and scalability almost for free.

The pattern scales from small projects (SQLite on your laptop) to production systems (PostgreSQL/Redis with multiple workers). Start simple, scale when needed.

Most importantly: your dataset IS your queue. The processing becomes operational, observable, and debuggable. You can always answer "what's the state of my data?" by querying the queue.

Try it for your next data processing project. Your future self will thank you when you need to debug why item #47,382 failed, or when you want to add a new processing stage without reprocessing everything.

## Resources

- [Source code for this project](https://github.com/rom1504/queue_as_dataset)
- [SQLite documentation](https://www.sqlite.org/docs.html)
- [FastAPI for building the API](https://fastapi.tiangolo.com/)
- [Similar patterns: Celery, BullMQ, RQ (Redis Queue)]

---

*This blogpost describes the architecture implemented in the queue_as_dataset project. See the README for setup instructions and examples.*
