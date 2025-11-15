"""FastAPI backend for queue management."""

import csv
import io
import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import QueueItemCreate, QueueStats
from .queue import QueueItem, SQLiteQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Queue Processor API",
    description="API for managing web page processing queue",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize queues
DB_PATH = "queue.db"
QUEUE_NAMES = ["page_queue", "stats_queue", "ml_queue"]

# Cache queue instances
_queues = {}

def get_queue(queue_name: str) -> SQLiteQueue:
    """Get or create a queue instance."""
    if queue_name not in _queues:
        _queues[queue_name] = SQLiteQueue(db_path=DB_PATH, queue_name=queue_name)
    return _queues[queue_name]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Queue Processor API",
        "version": "1.0.0",
        "queues": QUEUE_NAMES,
        "endpoints": {
            "queues": "/api/queues",
            "stats": "/api/stats?queue=<queue_name>",
            "speed": "/api/speed?queue=<queue_name>",
            "items": "/api/items?queue=<queue_name>",
            "add": "/api/items?queue=<queue_name>",
            "upload": "/api/upload?queue=<queue_name>",
        }
    }


@app.get("/api/queues")
async def get_queues():
    """Get list of available queues with their stats."""
    result = []
    for queue_name in QUEUE_NAMES:
        queue = get_queue(queue_name)
        stats = queue.get_stats()
        result.append({
            "name": queue_name,
            "stats": stats
        })
    return {"queues": result}



@app.get("/api/stats", response_model=QueueStats)
async def get_stats(queue: str = "page_queue"):
    """Get queue statistics.

    Args:
        queue: Queue name (default: page_queue)
    """
    if queue not in QUEUE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Must be one of: {', '.join(QUEUE_NAMES)}"
        )

    queue_instance = get_queue(queue)
    stats = queue_instance.get_stats()
    return QueueStats(**stats)


@app.get("/api/speed")
async def get_processing_speed(queue_name: str = "page_queue"):
    """Get current processing speed metrics.

    Args:
        queue_name: Queue name (default: page_queue)
    """
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Must be one of: {', '.join(QUEUE_NAMES)}"
        )

    queue_instance = get_queue(queue_name)
    conn = queue_instance._get_connection()
    current_time = time.time()

    # Get items completed in the last 60 seconds
    last_minute_query = f"""
        SELECT COUNT(*) as count, MIN(created_at) as first_time, MAX(created_at) as last_time
        FROM {queue_instance.queue_name}
        WHERE status = 'completed'
        AND json_extract(metadata, '$.completed_at') > ?
    """

    cursor = conn.execute(last_minute_query, (current_time - 60,))
    last_minute = cursor.fetchone()

    # Get items completed in the last 5 minutes for a more stable average
    last_5min_query = f"""
        SELECT COUNT(*) as count, MIN(created_at) as first_time, MAX(created_at) as last_time
        FROM {queue_instance.queue_name}
        WHERE status = 'completed'
        AND json_extract(metadata, '$.completed_at') > ?
    """

    cursor = conn.execute(last_5min_query, (current_time - 300,))
    last_5min = cursor.fetchone()

    # Calculate speeds
    speed_1min = 0
    speed_5min = 0
    eta_seconds = None

    if last_minute["count"] > 0:
        speed_1min = last_minute["count"] * 60 / 60  # items per minute

    if last_5min["count"] > 0:
        speed_5min = last_5min["count"] * 60 / 300  # items per minute

    # Get current stats for ETA
    stats = queue_instance.get_stats()
    pending = stats.get("pending", 0)
    processing = stats.get("processing", 0)
    remaining = pending + processing

    # Calculate ETA using 5-minute average (more stable)
    if speed_5min > 0 and remaining > 0:
        eta_seconds = (remaining / speed_5min) * 60

    return {
        "queue": queue_name,
        "speed_1min": round(speed_1min, 2),
        "speed_5min": round(speed_5min, 2),
        "items_last_1min": last_minute["count"],
        "items_last_5min": last_5min["count"],
        "remaining_items": remaining,
        "eta_seconds": round(eta_seconds) if eta_seconds else None,
        "eta_minutes": round(eta_seconds / 60, 1) if eta_seconds else None,
    }


@app.get("/api/items")
async def get_items(
    queue_name: str = "page_queue",
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get items from the queue.

    Args:
        queue_name: Queue name (default: page_queue)
        status: Filter by status (pending, processing, completed, failed)
        limit: Maximum number of items to return
        offset: Offset for pagination
    """
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Must be one of: {', '.join(QUEUE_NAMES)}"
        )

    if status and status not in ["pending", "processing", "completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid status. Must be one of: pending, processing, completed, failed"
        )

    queue_instance = get_queue(queue_name)
    items = queue_instance.get_items(status=status, limit=limit, offset=offset)
    return {
        "queue": queue_name,
        "items": items,
        "count": len(items),
        "limit": limit,
        "offset": offset,
    }


@app.post("/api/items")
async def add_item(item: QueueItemCreate, queue_name: str = "page_queue"):
    """Add a single item to the queue.

    Args:
        item: QueueItemCreate with page_url
        queue_name: Queue name (default: page_queue)
    """
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Must be one of: {', '.join(QUEUE_NAMES)}"
        )

    queue_instance = get_queue(queue_name)
    queue_item = QueueItem(
        payload={"page_url": str(item.page_url)},
        metadata={"source": "api"},
    )

    item_id = queue_instance.push(queue_item)

    logger.info(f"Added item {item_id} to {queue_name}: {item.page_url}")

    return {
        "id": item_id,
        "queue": queue_name,
        "page_url": str(item.page_url),
        "message": "Item added to queue"
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), queue_name: str = "page_queue"):
    """Upload a CSV file with URLs to add to the queue.

    CSV should have a 'url' column.

    Args:
        file: CSV file to upload
        queue_name: Queue name (default: page_queue)
    """
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Must be one of: {', '.join(QUEUE_NAMES)}"
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV"
        )

    try:
        queue_instance = get_queue(queue_name)

        # Read CSV content
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        csv_reader = csv.DictReader(csv_file)

        # Check for 'url' column
        if 'url' not in csv_reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail="CSV must have a 'url' column"
            )

        # Add items to queue
        added = 0
        errors = []

        for i, row in enumerate(csv_reader, start=1):
            url = row.get('url', '').strip()
            if not url:
                errors.append(f"Row {i}: Missing URL")
                continue

            try:
                queue_item = QueueItem(
                    payload={"page_url": url},
                    metadata={"source": "csv_upload", "row": i},
                )
                queue_instance.push(queue_item)
                added += 1
            except Exception as e:
                errors.append(f"Row {i} ({url}): {str(e)}")

        logger.info(f"CSV upload to {queue_name}: added {added} items, {len(errors)} errors")

        return {
            "queue": queue_name,
            "added": added,
            "errors": errors,
            "message": f"Added {added} items to {queue_name}"
        }

    except Exception as e:
        logger.error(f"Error processing CSV: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV: {str(e)}"
        )


@app.delete("/api/items/completed")
async def clear_completed(queue_name: str = "page_queue", older_than_days: int = 7):
    """Clear completed items older than specified days.

    Args:
        queue_name: Queue name (default: page_queue)
        older_than_days: Remove completed items older than this many days
    """
    if queue_name not in QUEUE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Must be one of: {', '.join(QUEUE_NAMES)}"
        )

    queue_instance = get_queue(queue_name)
    queue_instance.clear_completed(older_than_days=older_than_days)
    return {
        "queue": queue_name,
        "message": f"Cleared completed items older than {older_than_days} days"
    }


# Serve frontend
@app.get("/ui")
async def serve_frontend():
    """Serve the frontend UI."""
    return FileResponse("frontend/index.html")


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
