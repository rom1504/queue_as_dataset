"""Integration tests for the full queue processing pipeline."""

import tempfile
from pathlib import Path

import pytest

from queue_processor.models import InterleavedChunk, ProcessedPage
from queue_processor.queue import QueueItem, SQLiteQueue
from queue_processor.worker import QueueWorker, WebPageProcessor


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_queue.db"
        yield str(db_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_extract_multiple_text_and_image_chunks():
    """Test extraction of interleaved text and image chunks from complex HTML."""
    processor = WebPageProcessor()

    html = """
    <html>
    <body>
        <main>
            <h1>Article Title</h1>
            <p>This is the first paragraph with substantial text content that should be extracted
               as a chunk. It has enough content to pass the minimum length filter.</p>

            <p>Second paragraph also with enough text to be extracted as a separate chunk.
               This helps verify that we're getting multiple text chunks.</p>

            <figure>
                <img src="https://example.com/image1.jpg" width="800" height="600">
                <figcaption>This is a caption for the first image with enough text to be extracted</figcaption>
            </figure>

            <p>Third paragraph that comes after the first image. This should appear after
               the image chunk in the interleaved output structure.</p>

            <div>
                <img src="https://example.com/image2.jpg" width="600" height="400">
            </div>

            <p>Fourth paragraph after the second image with sufficient text content to ensure
               proper extraction and ordering of chunks.</p>

            <section>
                <h2>Subsection Title</h2>
                <p>Fifth paragraph in a subsection with enough text to be extracted as well.
                   This tests that we recurse properly into container elements.</p>

                <img src="https://example.com/image3.jpg" width="500" height="500">

                <p>Sixth paragraph after the third image, still within the subsection container.
                   This helps test nested structure extraction.</p>
            </section>
        </main>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Verify we got multiple chunks
    assert len(chunks) > 5

    # Count chunks by type
    text_chunks = [c for c in chunks if c.type == "text"]
    image_chunks = [c for c in chunks if c.type == "image"]

    # Should have multiple text chunks
    assert len(text_chunks) >= 5, f"Expected at least 5 text chunks, got {len(text_chunks)}"

    # Should have multiple image chunks
    assert len(image_chunks) >= 3, f"Expected at least 3 image chunks, got {len(image_chunks)}"

    # Verify images have absolute URLs
    for img_chunk in image_chunks:
        assert img_chunk.value.startswith("https://example.com/")

    # Verify interleaving - check that chunks alternate types at some point
    chunk_types = [c.type for c in chunks]
    # Should have both types appearing
    assert "text" in chunk_types
    assert "image" in chunk_types

    # Verify at least one transition from text to image or image to text
    has_transition = False
    for i in range(len(chunk_types) - 1):
        if chunk_types[i] != chunk_types[i + 1]:
            has_transition = True
            break
    assert has_transition, "Chunks should have type transitions for interleaved content"


def test_queue_metadata_persistence(temp_db, temp_output_dir):
    """Test that metadata is properly saved when acknowledging queue items."""
    queue = SQLiteQueue(db_path=temp_db, queue_name="test_queue")

    # Add item to queue
    item = QueueItem(
        payload={"page_url": "https://example.com"},
        metadata={"source": "test"},
    )
    item_id = queue.push(item)

    # Pop and acknowledge with updated metadata
    popped = queue.pop()
    assert popped is not None

    # Update metadata
    updated_metadata = popped.metadata.copy()
    updated_metadata["output_file"] = "data/processed_1.json"
    updated_metadata["num_chunks"] = 42
    updated_metadata["completed_at"] = 1234567890.0

    # Acknowledge with metadata
    queue.ack(popped.id, metadata=updated_metadata)

    # Retrieve and verify metadata was saved
    items = queue.get_items(status="completed")
    assert len(items) == 1
    assert items[0]["metadata"]["output_file"] == "data/processed_1.json"
    assert items[0]["metadata"]["num_chunks"] == 42
    assert items[0]["metadata"]["completed_at"] == 1234567890.0
    assert items[0]["metadata"]["source"] == "test"


def test_small_image_filtering():
    """Test that small images are filtered out."""
    processor = WebPageProcessor()

    html = """
    <html>
    <body>
        <main>
            <p>Some text content that should definitely be extracted from this test page.</p>

            <!-- Small images that should be filtered -->
            <img src="icon.png" width="20" height="20">
            <img src="button.png" width="30" height="30">
            <img src="tiny.jpg" width="10" height="10">

            <!-- Large images that should be kept -->
            <img src="photo1.jpg" width="800" height="600">
            <img src="photo2.jpg" width="1200" height="900">

            <p>More text content after the images to ensure proper extraction continues.</p>
        </main>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Get image chunks
    image_chunks = [c for c in chunks if c.type == "image"]

    # Should only have the large images
    assert len(image_chunks) == 2, f"Expected 2 images, got {len(image_chunks)}"

    # Verify correct images were kept
    image_urls = [c.value for c in image_chunks]
    assert any("photo1.jpg" in url for url in image_urls)
    assert any("photo2.jpg" in url for url in image_urls)
    assert not any("icon.png" in url for url in image_urls)
    assert not any("button.png" in url for url in image_urls)


def test_figure_element_extraction():
    """Test extraction of images from figure elements with captions."""
    processor = WebPageProcessor()

    html = """
    <html>
    <body>
        <main>
            <p>Introduction text that provides context for the article and sets up the figures below.
               This paragraph has been extended to ensure it meets the minimum length requirements
               for text chunk extraction in our processing pipeline.</p>

            <figure>
                <img src="/images/diagram.png" width="600" height="400">
                <figcaption>This is a detailed caption explaining the diagram shown above in the figure element.
                            Captions provide important context for understanding the visual content presented.</figcaption>
            </figure>

            <p>Text between figures that discusses the content and transitions to the next figure.
               This intermediate paragraph connects the two figures and provides narrative continuity
               between the different visual elements presented in the article.</p>

            <figure>
                <img src="/photos/example.jpg" width="800" height="600">
                <figcaption>Another caption with sufficient length to be extracted as a text chunk.
                            This figure caption demonstrates how we handle multiple figure elements.</figcaption>
            </figure>
        </main>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com/article")

    # Should have text and images
    text_chunks = [c for c in chunks if c.type == "text"]
    image_chunks = [c for c in chunks if c.type == "image"]

    assert len(image_chunks) == 2, f"Expected 2 images, got {len(image_chunks)}"
    assert len(text_chunks) >= 2, f"Expected at least 2 text chunks, got {len(text_chunks)}"

    # Verify image URLs are resolved to absolute
    assert image_chunks[0].value == "https://example.com/images/diagram.png"
    assert image_chunks[1].value == "https://example.com/photos/example.jpg"

    # Verify captions are extracted
    caption_texts = [c.value for c in text_chunks]
    assert any("detailed caption" in text for text in caption_texts)
    assert any("Another caption" in text for text in caption_texts)


def test_text_chunk_minimum_length():
    """Test that very short text chunks are filtered out."""
    processor = WebPageProcessor()

    html = """
    <html>
    <body>
        <main>
            <p>Short</p>
            <p>Also short</p>
            <p>This paragraph has enough text content to be extracted as a proper chunk
               and should appear in the final output.</p>
            <p>X</p>
            <p>Another substantial paragraph with sufficient text content to pass the
               minimum length filter and be included in the extraction results.</p>
        </main>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    text_chunks = [c for c in chunks if c.type == "text"]

    # Should only have the two long paragraphs
    assert len(text_chunks) == 2, f"Expected 2 text chunks, got {len(text_chunks)}"

    # Verify all chunks meet minimum length
    for chunk in text_chunks:
        assert len(chunk.value) > 100, f"Text chunk too short: {len(chunk.value)} chars"


def test_wikipedia_like_structure():
    """Test extraction from Wikipedia-like article structure."""
    processor = WebPageProcessor()

    html = """
    <html>
    <body>
        <main>
            <h1>Article Title</h1>
            <p>The main subject is a comprehensive topic that requires detailed explanation.
               This introduction provides an overview of the key concepts discussed.</p>

            <section>
                <h2>History</h2>
                <p>Historical context and background information about the subject matter.
                   This section provides important chronological details.</p>

                <figure>
                    <img src="//upload.example.org/history-image.jpg" width="250" height="150">
                </figure>

                <p>Additional historical details and analysis that follows the image.
                   This continues the narrative about the subject's history.</p>
            </section>

            <section>
                <h2>Description</h2>
                <p>Detailed description of physical characteristics and properties.
                   This section explains the key features in depth.</p>

                <aside>
                    <figure>
                        <img src="//upload.example.org/diagram.svg" width="200" height="200">
                        <figcaption>Scientific diagram showing the key structural elements of the subject.</figcaption>
                    </figure>
                </aside>

                <p>Further descriptive content that elaborates on the characteristics.
                   This provides additional detail and context.</p>
            </section>
        </main>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.org/wiki/Article")

    # Verify we got a good mix of content
    text_chunks = [c for c in chunks if c.type == "text"]
    image_chunks = [c for c in chunks if c.type == "image"]

    assert len(text_chunks) >= 4, f"Expected at least 4 text chunks, got {len(text_chunks)}"
    assert len(image_chunks) >= 2, f"Expected at least 2 images, got {len(image_chunks)}"

    # Verify protocol-relative URLs are handled
    for img_chunk in image_chunks:
        assert img_chunk.value.startswith("https://"), f"URL not properly resolved: {img_chunk.value}"

    # Verify images from nested structures are extracted
    image_urls = [c.value for c in image_chunks]
    assert any("history-image.jpg" in url for url in image_urls)
    assert any("diagram.svg" in url for url in image_urls)
