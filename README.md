# Queue Processor - Web Page to Interleaved Content

A prototype implementation of the "dataset as a queue" pattern for processing web pages into interleaved image/text content.

## Overview

This system implements a queue-based architecture for:
- Downloading web pages
- Extracting interleaved text and image content
- Storing results as JSON files
- Managing the processing pipeline via a web interface

## Architecture

The system consists of several components:

1. **SQLite Queue Backend** - Durable queue with visibility timeout and retry logic
2. **Worker** - Processes web pages into interleaved content
3. **FastAPI Backend** - REST API for queue management
4. **Frontend UI** - Web interface for monitoring and adding items
5. **Puller** - Example consumer that reads processed results

## Installation

This project uses `uv` for package management.

```bash
# Dependencies are already installed in .venv
# To activate the virtual environment:
source .venv/bin/activate

# Or run commands directly with uv:
uv run <command>
```

## Usage

### 1. Start the Backend API

```bash
uv run python -m queue_processor.api
```

The API will be available at `http://localhost:8000`

### 2. Start the Worker (in a separate terminal)

```bash
uv run python -m queue_processor.worker
```

The worker will poll the queue and process items as they become available.

### 3. Access the Web UI

Open your browser to:
- Frontend UI: `http://localhost:8000/ui`
- API docs: `http://localhost:8000/docs`

### 4. Add Items to the Queue

#### Via Web UI
1. Go to `http://localhost:8000/ui`
2. Use the "Add Item to Queue" form to add a single URL
3. Or upload a CSV file with a `url` column

#### Via API
```bash
curl -X POST http://localhost:8000/api/items \
  -H "Content-Type: application/json" \
  -d '{"page_url": "https://example.com"}'
```

#### Via CSV
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@example_urls.csv"
```

### 5. Use the Puller to View Results

```bash
# View summary of all completed items
uv run python -m queue_processor.puller

# View details of a specific item
uv run python -m queue_processor.puller --item-id 1
```

### 6. (Optional) Use Apache Beam for Parallel Processing

For faster batch processing, use the Beam worker with multiple parallel workers:

```bash
# Process with 4 parallel workers
uv run python -m queue_processor.beam_worker --workers 4

# Process with 8 workers for even faster throughput
uv run python -m queue_processor.beam_worker --workers 8
```

See [Beam Performance Report](beam_performance_report.md) for detailed benchmark results showing **3.5x speedup** with 4 workers.

## Queue Workflow

```
+----------------+
| Add URLs       |
| (UI/API/CSV)   |
+-------+--------+
        |
        v
+-------+--------+
| Queue          |
| (SQLite)       |
+-------+--------+
        |
        v
+-------+--------+
| Worker         |
| - Download     |
| - Extract      |
| - Process      |
+-------+--------+
        |
        v
+-------+--------+
| JSON Files     |
| (data/)        |
+-------+--------+
        |
        v
+-------+--------+
| Puller         |
| (Consumer)     |
+----------------+
```

## Data Format

Processed pages are saved as JSON files with the following structure:

```json
{
  "page_url": "https://example.com/article",
  "chunks": [
    {
      "type": "text",
      "value": "This is a paragraph of text content..."
    },
    {
      "type": "image",
      "value": "https://example.com/images/photo.jpg"
    },
    {
      "type": "text",
      "value": "More text content follows the image..."
    }
  ],
  "metadata": {
    "processed_at": 1731640000,
    "num_chunks": 3,
    "num_images": 1,
    "num_text": 2
  }
}
```

## API Endpoints

- `GET /` - API info
- `GET /api/stats` - Queue statistics
- `GET /api/items` - Get queue items (supports filtering and pagination)
- `POST /api/items` - Add a single item
- `POST /api/upload` - Upload CSV with URLs
- `DELETE /api/items/completed` - Clear old completed items
- `GET /ui` - Web interface

## Configuration

### Worker Options

```bash
uv run python -m queue_processor.worker \
  --db queue.db \
  --queue page_queue \
  --output-dir data \
  --poll-interval 5
```

### Puller Options

```bash
uv run python -m queue_processor.puller \
  --db queue.db \
  --queue page_queue \
  --output-dir data \
  --item-id 123
```

### Beam Worker Options

```bash
uv run python -m queue_processor.beam_worker \
  --db queue.db \
  --queue page_queue \
  --output-dir data \
  --workers 4 \
  --runner DirectRunner
```

Supported runners:
- `DirectRunner` - Local multi-threaded execution (default)
- `DataflowRunner` - Google Cloud Dataflow (requires cloud setup)
- `FlinkRunner` - Apache Flink (requires Flink cluster)

## Testing

Run the test suite:

```bash
uv run pytest tests/ -v
```

Tests cover:
- Queue operations (push, pop, ack, nack, fail)
- Visibility timeout and retry logic
- Web page content extraction
- Image filtering and URL resolution
- Text merging and cleanup

## Project Structure

```
.
├── queue_processor/
│   ├── __init__.py
│   ├── queue.py        # SQLite queue implementation
│   ├── models.py       # Pydantic data models
│   ├── worker.py       # Web page processor
│   ├── beam_worker.py  # Apache Beam parallel processor
│   ├── api.py          # FastAPI backend
│   └── puller.py       # Example consumer
├── frontend/
│   └── index.html      # Web UI
├── tests/
│   ├── test_queue.py   # Queue tests
│   └── test_worker.py  # Worker tests
├── data/               # Processed output files
├── queue.db            # SQLite database (created on first run)
└── example_urls.csv    # Example CSV file
```

## Features

- **Durable Queue**: SQLite-backed queue with ACID guarantees
- **Visibility Timeout**: Prevents multiple workers from processing the same item
- **Retry Logic**: Failed items can be retried automatically
- **Status Tracking**: Items can be pending, processing, completed, or failed
- **Parallel Processing**: Apache Beam integration for 3.5x speedup with 4 workers
- **Web Interface**: Easy monitoring and management
- **CSV Upload**: Bulk add URLs from CSV files
- **Interleaved Content**: Preserves document structure with text and images in order
- **Smart Filtering**: Removes navigation, scripts, tracking pixels, and small images
- **Relative URL Resolution**: Converts relative image URLs to absolute URLs

## Example Workflow

1. Start the backend:
   ```bash
   uv run python -m queue_processor.api
   ```

2. Start the worker in another terminal:
   ```bash
   uv run python -m queue_processor.worker
   ```

3. Add some URLs via the web UI at `http://localhost:8000/ui`

4. Watch the worker process them in real-time

5. Check the results:
   ```bash
   uv run python -m queue_processor.puller
   ```

6. Inspect a specific processed file:
   ```bash
   cat data/processed_1.json | jq
   ```

## Implementation Details

### Queue Backend

Based on the "dataset as a queue" pattern described in `dataset_as_queue.md`, the SQLite queue provides:

- **Push/Pop semantics**: Standard queue operations
- **Metadata tracking**: Each item stores producer, lineage, and processing info
- **Priority support**: Items can be ordered and filtered
- **Garbage collection**: Old completed items can be cleaned up

### Worker

The worker:
1. Polls the queue for pending items
2. Downloads the HTML page
3. Extracts interleaved text and image content
4. Saves results to JSON files
5. Updates queue metadata with output file reference
6. Acknowledges successful processing

### Content Extraction

The processor:
- Prioritizes main content areas (article, main tags)
- Removes navigation, footer, scripts, and styles
- Filters out tracking pixels and small images
- Merges consecutive text chunks
- Resolves relative URLs to absolute
- Maintains document order with interleaved chunks

## Performance

### Benchmark: 99 Wikipedia Pages

To test the system's throughput and quality, we processed 99 diverse Wikipedia articles:

**Processing Speed:**
- Total time: 2.11 minutes
- Throughput: **~46 pages/minute**
- Average per page: 1.31 seconds
- Success rate: 97% (3 failed disambiguation pages)

**Content Extraction Quality:**
- Total chunks extracted: 7,914
- Text chunks: 6,738 (85%)
- Image chunks: 1,176 (15%)
- Average per page: 81.6 chunks
- Average images per page: 12.1

**Top Content-Rich Pages:**
- Artificial Intelligence: 202 chunks
- Sculpture: 188 chunks
- Psychology: 179 chunks
- Economics: 164 chunks

The system successfully processes Wikipedia articles at high speed while maintaining quality interleaved extraction of text and images.

### Beam Parallel Processing

With Apache Beam integration, processing speed increases significantly:

**Test: 35 Wikipedia Pages**
- **Beam (4 workers)**: 13.02 seconds total, 161.3 pages/minute
- **Single-threaded**: Would take ~46 seconds, 46 pages/minute
- **Speedup**: **3.5x faster** with 4 parallel workers

The Beam implementation achieves 87.5% parallel efficiency, showing excellent scaling for I/O-bound web scraping workloads. See [beam_performance_report.md](beam_performance_report.md) for full details.

## License

This is a prototype implementation for demonstration purposes.
