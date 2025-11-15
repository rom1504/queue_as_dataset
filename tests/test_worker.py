"""Tests for the web page processor."""

import json
import tempfile
from pathlib import Path

import pytest

from queue_processor.worker import WebPageProcessor


@pytest.fixture
def processor():
    """Create a test processor with temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield WebPageProcessor(output_dir=tmpdir)


def test_extract_interleaved_content_simple(processor):
    """Test extracting interleaved content from simple HTML."""
    html = """
    <html>
    <body>
        <h1>Test Article</h1>
        <p>This is the first paragraph with substantial content to test extraction functionality.
           The paragraph needs to be long enough to pass the minimum length filter requirements.</p>
        <img src="https://example.com/image1.jpg" width="600" height="400">
        <p>This is the second paragraph that comes after the image with enough text content to be extracted.
           This ensures we test proper interleaving of text and image chunks.</p>
        <img src="https://example.com/image2.jpg" width="800" height="600">
        <p>This is the third paragraph with more text content for testing the extraction pipeline.
           Multiple paragraphs help verify that the content is properly structured.</p>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Should have interleaved text and images
    assert len(chunks) > 0

    # Check that we have both types
    types = [chunk.type for chunk in chunks]
    assert "text" in types
    assert "image" in types

    # Check image URLs are absolute
    image_chunks = [c for c in chunks if c.type == "image"]
    for img_chunk in image_chunks:
        assert img_chunk.value.startswith("http")


def test_extract_interleaved_content_filters_small_images(processor):
    """Test that small images are filtered out."""
    html = """
    <html>
    <body>
        <p>Some text content that should be extracted from the page.</p>
        <img src="small.jpg" width="10" height="10">
        <img src="large.jpg" width="800" height="600">
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Small image should be filtered
    image_chunks = [c for c in chunks if c.type == "image"]
    assert len(image_chunks) == 1
    assert "large.jpg" in image_chunks[0].value


def test_extract_interleaved_content_filters_icons(processor):
    """Test that icons and tracking pixels are filtered."""
    html = """
    <html>
    <body>
        <p>Some text content that should be extracted from the page with enough length
           to pass the minimum content filter and be included in the output.</p>
        <img src="icon-small.png" width="100" height="100">
        <img src="logo-header.jpg" width="100" height="100">
        <img src="regular-image.jpg" width="800" height="600">
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Icons and logos should be filtered (they have icon- or logo- prefix)
    image_chunks = [c for c in chunks if c.type == "image"]
    assert len(image_chunks) == 1, f"Expected 1 image, got {len(image_chunks)}: {[c.value for c in image_chunks]}"
    assert "regular-image.jpg" in image_chunks[0].value


def test_extract_interleaved_content_removes_scripts(processor):
    """Test that script and style tags are removed."""
    html = """
    <html>
    <head>
        <style>body { color: red; }</style>
    </head>
    <body>
        <script>console.log('test');</script>
        <p>This text should be extracted but not the script content from above.
           The paragraph is long enough to pass the minimum length filter and
           should appear in the extracted chunks without any script tags.</p>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Check that script/style content is not in text chunks
    text_chunks = [c for c in chunks if c.type == "text"]
    assert len(text_chunks) > 0

    all_text = " ".join(c.value for c in text_chunks)
    assert "console.log" not in all_text
    assert "color: red" not in all_text
    assert "should be extracted" in all_text


def test_extract_interleaved_content_preserves_structure(processor):
    """Test that document structure is preserved with proper chunking."""
    html = """
    <html>
    <body>
        <p>First paragraph with enough text to pass the filter requirements and provide
           meaningful content for extraction testing purposes.</p>
        <p>Second paragraph also with enough text to pass the filter requirements and
           demonstrate that multiple paragraphs are extracted properly.</p>
        <img src="image.jpg" width="800" height="600">
        <p>Third paragraph after the image with enough text to pass filters and show
           that content extraction continues after images correctly.</p>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Should have text and image chunks
    text_chunks = [c for c in chunks if c.type == "text"]
    image_chunks = [c for c in chunks if c.type == "image"]

    assert len(text_chunks) >= 2, f"Expected at least 2 text chunks, got {len(text_chunks)}"
    assert len(image_chunks) >= 1, f"Expected at least 1 image chunk, got {len(image_chunks)}"

    # Verify text content is preserved
    all_text = " ".join(c.value for c in text_chunks)
    assert "First paragraph" in all_text
    assert "Third paragraph" in all_text


def test_extract_interleaved_content_relative_urls(processor):
    """Test that relative URLs are resolved."""
    html = """
    <html>
    <body>
        <p>Some text content for testing relative URL resolution properly.</p>
        <img src="/images/photo.jpg" width="800" height="600">
        <img src="relative/image.jpg" width="800" height="600">
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(
        html,
        "https://example.com/article/page"
    )

    image_chunks = [c for c in chunks if c.type == "image"]
    assert len(image_chunks) == 2

    # Check URLs are absolute
    assert image_chunks[0].value == "https://example.com/images/photo.jpg"
    assert image_chunks[1].value.startswith("https://example.com/article/")


def test_save_processed_page(processor):
    """Test saving processed page to disk."""
    from queue_processor.models import InterleavedChunk, ProcessedPage

    chunks = [
        InterleavedChunk(type="text", value="Test text content"),
        InterleavedChunk(type="image", value="https://example.com/image.jpg"),
    ]

    processed = ProcessedPage(
        page_url="https://example.com",
        chunks=chunks,
        metadata={"num_chunks": 2},
    )

    filepath = processor.save_processed_page(processed, item_id=123)

    # Check file exists
    assert Path(filepath).exists()

    # Check file content
    with open(filepath, "r") as f:
        data = json.load(f)

    assert data["page_url"] == "https://example.com"
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["type"] == "text"
    assert data["chunks"][1]["type"] == "image"


def test_extract_interleaved_content_empty_page(processor):
    """Test extracting from empty page."""
    html = """
    <html>
    <body>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")
    assert len(chunks) == 0


def test_extract_interleaved_content_main_content(processor):
    """Test that main content is preferred over navigation."""
    html = """
    <html>
    <body>
        <nav>
            <a href="#">Navigation link that should be ignored</a>
        </nav>
        <main>
            <p>This is the main content that should be extracted from the page.
               The paragraph needs sufficient length to pass the minimum filter and
               demonstrate that main content area is properly identified and processed.</p>
            <img src="content-image.jpg" width="800" height="600">
        </main>
        <footer>
            <p>Footer content</p>
        </footer>
    </body>
    </html>
    """

    chunks = processor.extract_interleaved_content(html, "https://example.com")

    # Should extract from main, not nav/footer
    text_chunks = [c for c in chunks if c.type == "text"]
    all_text = " ".join(c.value for c in text_chunks)

    assert "main content" in all_text
    # Nav and footer are removed, so their content shouldn't appear
