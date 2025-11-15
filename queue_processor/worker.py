"""Worker for processing web pages into interleaved content."""

import json
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from .models import InterleavedChunk, ProcessedPage
from .queue import QueueItem, SQLiteQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebPageProcessor:
    """Processes web pages into interleaved text/image content."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download_page(self, url: str, timeout: int = 30) -> Optional[str]:
        """Download HTML content from a URL.

        Args:
            url: URL to download
            timeout: Request timeout in seconds

        Returns:
            HTML content as string, or None if failed
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def extract_interleaved_content(
        self, html: str, base_url: str
    ) -> list[InterleavedChunk]:
        """Extract interleaved text and image content from HTML.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative URLs

        Returns:
            List of InterleavedChunk objects
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        chunks = []

        # Find the main content area (try common patterns)
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=lambda x: x and "content" in x.lower() if x else False)
            or soup.find("body")
        )

        if not main_content:
            main_content = soup.find("body")

        if not main_content:
            return chunks

        # Extract content by walking through elements in order
        self._extract_from_element(main_content, base_url, chunks)

        # Filter out very short text chunks but keep all images
        final_chunks = [
            chunk for chunk in chunks
            if chunk.type == "image" or (chunk.type == "text" and len(chunk.value) > 100)
        ]

        return final_chunks

    def _extract_from_element(
        self, element: Tag, base_url: str, chunks: list[InterleavedChunk]
    ):
        """Recursively extract content from an element.

        Args:
            element: BeautifulSoup element
            base_url: Base URL for resolving relative URLs
            chunks: List to append chunks to
        """
        for child in element.children:
            if isinstance(child, NavigableString):
                # Skip pure whitespace
                continue

            if not isinstance(child, Tag):
                continue

            # Handle images - add immediately and don't recurse
            if child.name == "img":
                src = child.get("src")
                if src:
                    absolute_url = urljoin(base_url, src)
                    if self._is_valid_image(child, src):
                        chunks.append(
                            InterleavedChunk(type="image", value=absolute_url)
                        )
                continue

            # Handle text block elements
            if child.name in ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]:
                # Extract images from within this block first
                imgs = child.find_all("img")

                if imgs:
                    # If block contains images, process it more carefully
                    # Get text before first image
                    text_parts = []
                    for part in child.children:
                        if isinstance(part, Tag) and part.name == "img":
                            # Flush accumulated text
                            if text_parts:
                                text = " ".join(text_parts).strip()
                                if len(text) > 50:
                                    chunks.append(InterleavedChunk(type="text", value=text))
                                text_parts = []

                            # Add image
                            src = part.get("src")
                            if src:
                                absolute_url = urljoin(base_url, src)
                                if self._is_valid_image(part, src):
                                    chunks.append(
                                        InterleavedChunk(type="image", value=absolute_url)
                                    )
                        else:
                            # Accumulate text
                            if isinstance(part, NavigableString):
                                text_parts.append(str(part))
                            elif isinstance(part, Tag):
                                text_parts.append(part.get_text(" "))

                    # Flush any remaining text
                    if text_parts:
                        text = " ".join(text_parts).strip()
                        if len(text) > 50:
                            chunks.append(InterleavedChunk(type="text", value=text))
                else:
                    # No images in this block, just get the text
                    text = child.get_text(" ", strip=True)
                    if len(text) > 50:
                        chunks.append(InterleavedChunk(type="text", value=text))

            # Handle figure elements (often contain images)
            elif child.name == "figure":
                # Extract image from figure
                img = child.find("img")
                if img:
                    src = img.get("src")
                    if src:
                        absolute_url = urljoin(base_url, src)
                        if self._is_valid_image(img, src):
                            chunks.append(
                                InterleavedChunk(type="image", value=absolute_url)
                            )
                # Also extract caption if present
                caption = child.find("figcaption")
                if caption:
                    text = caption.get_text(" ", strip=True)
                    if len(text) > 50:
                        chunks.append(InterleavedChunk(type="text", value=text))

            # Handle containers - recurse into them
            elif child.name in ["div", "section", "article", "main", "aside", "table", "tbody", "tr", "td"]:
                self._extract_from_element(child, base_url, chunks)

    def _is_valid_image(self, img_element: Tag, src: str) -> bool:
        """Check if an image is worth including.

        Args:
            img_element: BeautifulSoup img tag
            src: Image source URL

        Returns:
            True if image should be included
        """
        # Skip tracking pixels and very small images
        width = img_element.get("width")
        height = img_element.get("height")

        if width and height:
            try:
                w = int(width)
                h = int(height)
                # Skip really small images (likely icons/buttons)
                if w < 40 or h < 40:
                    return False
                # Skip images that are clearly tiny
                if w * h < 2000:  # e.g., 40x50 or smaller
                    return False
            except ValueError:
                pass

        # Skip common icon/logo patterns (but be less aggressive)
        src_lower = src.lower()
        skip_patterns = [
            "icon-", "logo-", "avatar", "emoji",
            "tracking", "analytics", "ad-banner"
        ]

        if any(pattern in src_lower for pattern in skip_patterns):
            return False

        # Skip data URLs (usually small icons)
        if src.startswith("data:"):
            return False

        # Skip SVG icons (usually interface elements)
        if src.endswith('.svg') and any(s in src_lower for s in ['icon', 'symbol', 'button']):
            return False

        return True

    def process_page(self, url: str) -> Optional[ProcessedPage]:
        """Process a web page into interleaved content.

        Args:
            url: URL to process

        Returns:
            ProcessedPage object or None if failed
        """
        logger.info(f"Processing: {url}")

        html = self.download_page(url)
        if not html:
            return None

        chunks = self.extract_interleaved_content(html, url)

        if not chunks:
            logger.warning(f"No content extracted from {url}")
            return None

        return ProcessedPage(
            page_url=url,
            chunks=chunks,
            metadata={
                "processed_at": time.time(),
                "num_chunks": len(chunks),
                "num_images": sum(1 for c in chunks if c.type == "image"),
                "num_text": sum(1 for c in chunks if c.type == "text"),
            }
        )

    def save_processed_page(self, processed: ProcessedPage, item_id: int) -> str:
        """Save processed page to disk.

        Args:
            processed: ProcessedPage object
            item_id: Queue item ID

        Returns:
            Path to saved file
        """
        filename = f"processed_{item_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(processed.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed page to {filepath}")
        return str(filepath)


class QueueWorker:
    """Worker that processes items from the queue."""

    def __init__(
        self,
        queue: SQLiteQueue,
        processor: WebPageProcessor,
        poll_interval: int = 5,
        visibility_timeout: int = 300,
    ):
        self.queue = queue
        self.processor = processor
        self.poll_interval = poll_interval
        self.visibility_timeout = visibility_timeout
        self.running = False

    def run(self):
        """Run the worker loop."""
        self.running = True
        logger.info("Worker started")

        try:
            while self.running:
                item = self.queue.pop(visibility_timeout=self.visibility_timeout)

                if item is None:
                    # No items available, sleep and retry
                    time.sleep(self.poll_interval)
                    continue

                self.process_item(item)

        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
        finally:
            self.running = False

    def process_item(self, item: QueueItem):
        """Process a single queue item.

        Args:
            item: QueueItem to process
        """
        url = item.payload.get("page_url")
        if not url:
            logger.error(f"Item {item.id} has no page_url")
            self.queue.fail(item.id, "Missing page_url in payload")
            return

        try:
            # Process the page
            processed = self.processor.process_page(url)

            if processed is None:
                self.queue.fail(item.id, "Failed to process page")
                return

            # Save to disk
            filepath = self.processor.save_processed_page(processed, item.id)

            # Update metadata with result
            item.metadata["output_file"] = filepath
            item.metadata["num_chunks"] = len(processed.chunks)
            item.metadata["completed_at"] = time.time()

            # Acknowledge success with updated metadata
            self.queue.ack(item.id, metadata=item.metadata)
            logger.info(
                f"Successfully processed item {item.id}: {url} "
                f"({len(processed.chunks)} chunks)"
            )

        except Exception as e:
            logger.error(f"Error processing item {item.id}: {e}", exc_info=True)
            self.queue.fail(item.id, str(e))

    def stop(self):
        """Stop the worker."""
        self.running = False


def main():
    """Main entry point for the worker."""
    import argparse

    parser = argparse.ArgumentParser(description="Queue worker for processing web pages")
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
        "--poll-interval",
        type=int,
        default=5,
        help="Polling interval in seconds"
    )

    args = parser.parse_args()

    queue = SQLiteQueue(db_path=args.db, queue_name=args.queue)
    processor = WebPageProcessor(output_dir=args.output_dir)
    worker = QueueWorker(queue, processor, poll_interval=args.poll_interval)

    logger.info(f"Starting worker with queue '{args.queue}' from {args.db}")
    worker.run()


if __name__ == "__main__":
    main()
