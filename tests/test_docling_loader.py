"""Tests for the DoclingLoader (PDF, DOCX, PPTX, XLSX, HTML, CSV, images)."""

import tempfile
from pathlib import Path

import pytest

from cyllama.rag.loaders import (
    DoclingLoader,
    PDFLoader,
    LoaderError,
    load_document,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDoclingLoader:
    """Test DoclingLoader class."""

    def test_pdf_loader_is_alias(self):
        """Test that PDFLoader is an alias for DoclingLoader."""
        assert PDFLoader is DoclingLoader

    def test_supported_extensions(self):
        """Test that all expected extensions are supported."""
        loader = DoclingLoader()
        expected = {
            ".pdf", ".docx", ".pptx", ".xlsx",
            ".html", ".htm", ".csv",
            ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
        }
        assert loader.SUPPORTED_EXTENSIONS == expected

    def test_unsupported_extension(self, temp_dir):
        """Test error for unsupported file type."""
        file_path = temp_dir / "test.rtf"
        file_path.write_text("some content")

        loader = DoclingLoader()
        with pytest.raises(LoaderError, match="Unsupported file type"):
            loader.load(file_path)

    def test_nonexistent_file(self):
        """Test error for nonexistent file."""
        loader = DoclingLoader()
        with pytest.raises(LoaderError, match="File not found"):
            loader.load("/nonexistent/document.pdf")

    def test_load_pdf(self):
        """Test loading a real PDF file if available."""
        pdf_path = Path("tests/data/test.pdf")
        if not pdf_path.exists():
            pytest.skip("No test PDF available")

        loader = DoclingLoader()
        docs = loader.load(pdf_path)

        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "pdf"
        assert len(docs[0].text) > 0

    def test_load_docx(self):
        """Test loading a DOCX file if available."""
        docx_path = Path("tests/data/test.docx")
        if not docx_path.exists():
            pytest.skip("No test DOCX available")

        loader = DoclingLoader()
        docs = loader.load(docx_path)

        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "docx"
        assert len(docs[0].text) > 0

    def test_load_pptx(self):
        """Test loading a PPTX file if available."""
        pptx_path = Path("tests/data/test.pptx")
        if not pptx_path.exists():
            pytest.skip("No test PPTX available")

        loader = DoclingLoader()
        docs = loader.load(pptx_path)

        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "pptx"
        assert len(docs[0].text) > 0

    def test_load_image(self):
        """Test loading an image file if available."""
        img_path = Path("tests/data/test.jpg")
        if not img_path.exists():
            pytest.skip("No test image available")

        loader = DoclingLoader(ocr=True)
        docs = loader.load(img_path)

        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "image"
        assert len(docs[0].text) > 0

    def test_load_html(self, temp_dir):
        """Test loading an HTML file."""
        html_path = temp_dir / "test.html"
        html_path.write_text("<html><body><h1>Title</h1><p>Content here.</p></body></html>")

        loader = DoclingLoader()
        docs = loader.load(html_path)

        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "html"
        assert docs[0].metadata["filename"] == "test.html"
        assert len(docs[0].text) > 0

    def test_load_csv(self, temp_dir):
        """Test loading a CSV file."""
        csv_path = temp_dir / "test.csv"
        csv_path.write_text("name,age,city\nAlice,30,London\nBob,25,Paris\n")

        loader = DoclingLoader()
        docs = loader.load(csv_path)

        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "csv"
        assert len(docs[0].text) > 0

    def test_metadata_fields(self, temp_dir):
        """Test that metadata contains expected fields."""
        html_path = temp_dir / "meta_test.html"
        html_path.write_text("<html><body><p>Test</p></body></html>")

        loader = DoclingLoader()
        docs = loader.load(html_path)

        assert "source" in docs[0].metadata
        assert "filename" in docs[0].metadata
        assert "filetype" in docs[0].metadata
        assert docs[0].metadata["filename"] == "meta_test.html"
        assert docs[0].id == str(html_path)


class TestConvenienceFunctionsDocling:
    """Test load_document convenience function with docling formats."""

    def test_load_document_html(self, temp_dir):
        """Test load_document with HTML file."""
        file_path = temp_dir / "test.html"
        file_path.write_text("<html><body><p>Hello</p></body></html>")

        docs = load_document(file_path)
        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "html"

    def test_load_document_csv(self, temp_dir):
        """Test load_document with CSV file."""
        file_path = temp_dir / "test.csv"
        file_path.write_text("a,b\n1,2\n")

        docs = load_document(file_path)
        assert len(docs) >= 1
        assert docs[0].metadata["filetype"] == "csv"

    def test_load_document_docx_routes_to_docling(self):
        """Test that .docx extension routes to DoclingLoader."""
        assert ".docx" in DoclingLoader.SUPPORTED_EXTENSIONS
