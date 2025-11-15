"""SQLite-backed queue implementation for dataset-as-queue pattern."""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class QueueItem:
    """Represents an item in the queue."""

    def __init__(
        self,
        id: Optional[int] = None,
        payload: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
        timestamp: Optional[float] = None,
    ):
        self.id = id
        self.payload = payload or {}
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueItem":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
        )


class SQLiteQueue:
    """SQLite-backed queue with visibility timeout and retry logic."""

    def __init__(self, db_path: str = "queue.db", queue_name: str = "default"):
        self.db_path = db_path
        self.queue_name = queue_name
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.queue_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    available_at INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    reserved INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)

            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.queue_name}_available
                ON {self.queue_name}(available_at, reserved, id)
            """)

            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.queue_name}_status
                ON {self.queue_name}(status)
            """)

            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def push(self, item: QueueItem) -> int:
        """Push an item to the queue.

        Args:
            item: QueueItem to push

        Returns:
            The ID of the inserted item
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"""
                INSERT INTO {self.queue_name}
                (available_at, payload, metadata, created_at, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    json.dumps(item.payload),
                    json.dumps(item.metadata),
                    item.timestamp,
                    "pending",
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def pop(self, visibility_timeout: int = 300) -> Optional[QueueItem]:
        """Pop an item from the queue with visibility timeout.

        Args:
            visibility_timeout: Seconds to hide the item from other consumers

        Returns:
            QueueItem if available, None otherwise
        """
        conn = self._get_connection()
        try:
            now = int(time.time())

            # Find available item (either not reserved or visibility timeout expired)
            cursor = conn.execute(
                f"""
                SELECT id, payload, metadata, created_at FROM {self.queue_name}
                WHERE (
                    (reserved = 0 AND status = 'pending')
                    OR (reserved = 1 AND available_at <= ? AND status = 'processing')
                  )
                  AND available_at <= ?
                ORDER BY id
                LIMIT 1
                """,
                (now, now),
            )

            row = cursor.fetchone()
            if not row:
                return None

            qid = row["id"]
            payload = json.loads(row["payload"])
            metadata = json.loads(row["metadata"])
            created_at = row["created_at"]

            # Reserve the item (or re-reserve if visibility timeout expired)
            conn.execute(
                f"""
                UPDATE {self.queue_name}
                SET reserved = 1,
                    attempts = attempts + 1,
                    available_at = ?,
                    status = 'processing'
                WHERE id = ?
                  AND (
                    (reserved = 0 AND status = 'pending')
                    OR (reserved = 1 AND available_at <= ?)
                  )
                """,
                (now + visibility_timeout, qid, now),
            )

            if conn.total_changes == 0:
                # Lost race condition
                conn.commit()
                return None

            conn.commit()

            return QueueItem(
                id=qid,
                payload=payload,
                metadata=metadata,
                timestamp=created_at,
            )
        finally:
            conn.close()

    def ack(self, item_id: int, metadata: dict[str, Any] = None):
        """Acknowledge successful processing of an item.

        Args:
            item_id: ID of the item to acknowledge
            metadata: Optional updated metadata to save
        """
        conn = self._get_connection()
        try:
            if metadata:
                conn.execute(
                    f"UPDATE {self.queue_name} SET status = 'completed', metadata = ? WHERE id = ?",
                    (json.dumps(metadata), item_id),
                )
            else:
                conn.execute(
                    f"UPDATE {self.queue_name} SET status = 'completed' WHERE id = ?",
                    (item_id,),
                )
            conn.commit()
        finally:
            conn.close()

    def nack(self, item_id: int):
        """Negative acknowledge - return item to queue for retry.

        Args:
            item_id: ID of the item to return
        """
        conn = self._get_connection()
        try:
            conn.execute(
                f"""
                UPDATE {self.queue_name}
                SET reserved = 0,
                    available_at = ?,
                    status = 'pending'
                WHERE id = ?
                """,
                (int(time.time()), item_id),
            )
            conn.commit()
        finally:
            conn.close()

    def fail(self, item_id: int, error: str = ""):
        """Mark an item as failed.

        Args:
            item_id: ID of the item
            error: Error message
        """
        conn = self._get_connection()
        try:
            # Update metadata with error
            cursor = conn.execute(
                f"SELECT metadata FROM {self.queue_name} WHERE id = ?",
                (item_id,),
            )
            row = cursor.fetchone()
            if row:
                metadata = json.loads(row["metadata"])
                metadata["error"] = error
                metadata["failed_at"] = time.time()

                conn.execute(
                    f"""
                    UPDATE {self.queue_name}
                    SET status = 'failed',
                        metadata = ?
                    WHERE id = ?
                    """,
                    (json.dumps(metadata), item_id),
                )
                conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"""
                SELECT
                    status,
                    COUNT(*) as count
                FROM {self.queue_name}
                GROUP BY status
                """
            )

            stats = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
            }

            for row in cursor:
                stats[row["status"]] = row["count"]

            # Total count
            cursor = conn.execute(f"SELECT COUNT(*) as total FROM {self.queue_name}")
            stats["total"] = cursor.fetchone()["total"]

            return stats
        finally:
            conn.close()

    def get_items(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get items from the queue.

        Args:
            status: Filter by status (pending, processing, completed, failed)
            limit: Maximum number of items to return
            offset: Offset for pagination

        Returns:
            List of queue items as dictionaries
        """
        conn = self._get_connection()
        try:
            if status:
                cursor = conn.execute(
                    f"""
                    SELECT id, payload, metadata, created_at, status, attempts
                    FROM {self.queue_name}
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                )
            else:
                cursor = conn.execute(
                    f"""
                    SELECT id, payload, metadata, created_at, status, attempts
                    FROM {self.queue_name}
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )

            items = []
            for row in cursor:
                items.append({
                    "id": row["id"],
                    "payload": json.loads(row["payload"]),
                    "metadata": json.loads(row["metadata"]),
                    "timestamp": row["created_at"],
                    "status": row["status"],
                    "attempts": row["attempts"],
                })

            return items
        finally:
            conn.close()

    def clear_completed(self, older_than_days: int = 7):
        """Clear completed items older than specified days.

        Args:
            older_than_days: Remove completed items older than this many days
        """
        conn = self._get_connection()
        try:
            cutoff = time.time() - (older_than_days * 24 * 60 * 60)
            conn.execute(
                f"""
                DELETE FROM {self.queue_name}
                WHERE status = 'completed' AND created_at < ?
                """,
                (cutoff,),
            )
            conn.commit()
        finally:
            conn.close()
