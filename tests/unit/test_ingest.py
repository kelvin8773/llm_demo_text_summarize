# tests/unit/test_ingest.py - Unit tests for document ingestion

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, mock_open
from utils.ingest import (
    load_document,
    _load_pdf,
    _load_txt,
    _load_docx
)
from tests.fixtures.sample_texts import (
    SAMPLE_PDF_CONTENT, SAMPLE_TXT_CONTENT, SAMPLE_DOCX_CONTENT,
    EMPTY_TEXT, VERY_SHORT_TEXT
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestLoadDocument:
    """Test cases for load_document function."""
    
    def test_load_pdf_file(self):
        """Test loading PDF file."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        with patch('utils.ingest._load_pdf') as mock_load_pdf:
            mock_load_pdf.return_value = SAMPLE_PDF_CONTENT
            
            result = load_document(mock_file)
            
            assert result == SAMPLE_PDF_CONTENT
            mock_load_pdf.assert_called_once_with(mock_file)
    
    def test_load_txt_file(self):
        """Test loading TXT file."""
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.size = 1024
        
        with patch('utils.ingest._load_txt') as mock_load_txt:
            mock_load_txt.return_value = SAMPLE_TXT_CONTENT
            
            result = load_document(mock_file)
            
            assert result == SAMPLE_TXT_CONTENT
            mock_load_txt.assert_called_once_with(mock_file)
    
    def test_load_docx_file(self):
        """Test loading DOCX file."""
        mock_file = Mock()
        mock_file.name = "test.docx"
        mock_file.size = 1024
        
        with patch('utils.ingest._load_docx') as mock_load_docx:
            mock_load_docx.return_value = SAMPLE_DOCX_CONTENT
            
            result = load_document(mock_file)
            
            assert result == SAMPLE_DOCX_CONTENT
            mock_load_docx.assert_called_once_with(mock_file)
    
    def test_unsupported_file_type(self):
        """Test loading unsupported file type."""
        mock_file = Mock()
        mock_file.name = "test.xyz"
        mock_file.size = 1024
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(mock_file)
    
    def test_no_file_provided(self):
        """Test loading with no file provided."""
        with pytest.raises(ValueError, match="No file provided"):
            load_document(None)
    
    def test_file_too_large(self):
        """Test loading file that's too large."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 15 * 1024 * 1024  # 15MB
        
        with pytest.raises(ValueError, match="File too large"):
            load_document(mock_file)
    
    def test_file_loading_error(self):
        """Test handling of file loading errors."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        with patch('utils.ingest._load_pdf') as mock_load_pdf:
            mock_load_pdf.side_effect = Exception("PDF loading failed")
            
            with pytest.raises(Exception, match="Failed to load document"):
                load_document(mock_file)
    
    def test_file_without_name_attribute(self):
        """Test loading file without name attribute."""
        mock_file = Mock()
        mock_file.size = 1024
        # No name attribute
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(mock_file)
    
    def test_file_without_size_attribute(self):
        """Test loading file without size attribute."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        # No size attribute
        
        with patch('utils.ingest._load_pdf') as mock_load_pdf:
            mock_load_pdf.return_value = SAMPLE_PDF_CONTENT
            
            result = load_document(mock_file)
            
            assert result == SAMPLE_PDF_CONTENT


class TestLoadPdf:
    """Test cases for _load_pdf function."""
    
    def test_successful_pdf_loading(self):
        """Test successful PDF loading."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            # Mock PDF reader and pages
            mock_reader = Mock()
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"
            mock_reader.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_reader
            
            result = _load_pdf(mock_file)
            
            assert result == "Page 1 content Page 2 content"
            mock_pdf_reader.assert_called_once_with(mock_file)
    
    def test_empty_pdf(self):
        """Test loading empty PDF."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_reader = Mock()
            mock_reader.pages = []
            mock_pdf_reader.return_value = mock_reader
            
            with pytest.raises(ValueError, match="PDF file appears to be empty"):
                _load_pdf(mock_file)
    
    def test_pdf_with_empty_pages(self):
        """Test loading PDF with empty pages."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = ""
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader
            
            with pytest.raises(ValueError, match="No readable text found"):
                _load_pdf(mock_file)
    
    def test_pdf_with_very_little_text(self):
        """Test loading PDF with very little text."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = "Short"
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader
            
            with pytest.raises(ValueError, match="PDF contains very little text"):
                _load_pdf(mock_file)
    
    def test_pdf_with_page_extraction_error(self):
        """Test handling of page extraction errors."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_reader = Mock()
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"
            mock_page2 = Mock()
            mock_page2.extract_text.side_effect = Exception("Page extraction failed")
            mock_reader.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_reader
            
            result = _load_pdf(mock_file)
            
            # Should still work with successful pages
            assert result == "Page 1 content"
    
    def test_password_protected_pdf(self):
        """Test handling of password-protected PDF."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("Password required")
            
            with pytest.raises(ValueError, match="PDF appears to be password-protected"):
                _load_pdf(mock_file)
    
    def test_corrupted_pdf(self):
        """Test handling of corrupted PDF."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("PDF is corrupted")
            
            with pytest.raises(ValueError, match="PDF file appears to be corrupted"):
                _load_pdf(mock_file)
    
    def test_general_pdf_error(self):
        """Test handling of general PDF errors."""
        mock_file = Mock()
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("General PDF error")
            
            with pytest.raises(Exception, match="Error reading PDF"):
                _load_pdf(mock_file)


class TestLoadTxt:
    """Test cases for _load_txt function."""
    
    def test_successful_txt_loading_utf8(self):
        """Test successful TXT loading with UTF-8 encoding."""
        mock_file = Mock()
        mock_file.read.return_value = SAMPLE_TXT_CONTENT.encode('utf-8')
        
        result = _load_txt(mock_file)
        
        assert result == SAMPLE_TXT_CONTENT
        assert mock_file.seek.call_count == 2  # Called twice for reset
    
    def test_successful_txt_loading_latin1(self):
        """Test successful TXT loading with Latin-1 encoding."""
        mock_file = Mock()
        mock_file.read.side_effect = [
            SAMPLE_TXT_CONTENT.encode('latin-1'),  # First read fails UTF-8
            SAMPLE_TXT_CONTENT.encode('latin-1')   # Second read succeeds Latin-1
        ]
        
        # Mock UnicodeDecodeError for first read
        with patch('builtins.decode', side_effect=[UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'), SAMPLE_TXT_CONTENT]):
            result = _load_txt(mock_file)
            
            assert result == SAMPLE_TXT_CONTENT
    
    def test_successful_txt_loading_cp1252(self):
        """Test successful TXT loading with CP1252 encoding."""
        mock_file = Mock()
        
        # Mock multiple encoding failures
        with patch('builtins.decode', side_effect=[
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'),
            UnicodeDecodeError('latin-1', b'', 0, 1, 'invalid start byte'),
            SAMPLE_TXT_CONTENT
        ]):
            result = _load_txt(mock_file)
            
            assert result == SAMPLE_TXT_CONTENT
    
    def test_empty_txt_file(self):
        """Test loading empty TXT file."""
        mock_file = Mock()
        mock_file.read.return_value = b""
        
        with pytest.raises(ValueError, match="Text file appears to be empty"):
            _load_txt(mock_file)
    
    def test_txt_file_with_very_little_content(self):
        """Test loading TXT file with very little content."""
        mock_file = Mock()
        mock_file.read.return_value = b"Short"
        
        with pytest.raises(ValueError, match="Text file appears to be empty"):
            _load_txt(mock_file)
    
    def test_txt_file_encoding_error(self):
        """Test handling of TXT file encoding errors."""
        mock_file = Mock()
        mock_file.read.return_value = b"Invalid bytes"
        
        # Mock all encoding attempts to fail
        with patch('builtins.decode', side_effect=UnicodeDecodeError('cp1252', b'', 0, 1, 'invalid start byte')):
            with pytest.raises(Exception, match="Error reading text file"):
                _load_txt(mock_file)


class TestLoadDocx:
    """Test cases for _load_docx function."""
    
    def test_successful_docx_loading(self):
        """Test successful DOCX loading."""
        mock_file = Mock()
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            # Mock document and paragraphs
            mock_doc = Mock()
            mock_para1 = Mock()
            mock_para1.text = "Paragraph 1 content"
            mock_para2 = Mock()
            mock_para2.text = "Paragraph 2 content"
            mock_para3 = Mock()
            mock_para3.text = ""  # Empty paragraph
            mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
            mock_docx.return_value = mock_doc
            
            result = _load_docx(mock_file)
            
            assert result == "Paragraph 1 content Paragraph 2 content"
            mock_docx.assert_called_once_with(mock_file)
    
    def test_empty_docx_file(self):
        """Test loading empty DOCX file."""
        mock_file = Mock()
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            mock_doc = Mock()
            mock_doc.paragraphs = []
            mock_docx.return_value = mock_doc
            
            with pytest.raises(ValueError, match="DOCX file appears to be empty"):
                _load_docx(mock_file)
    
    def test_docx_file_with_empty_paragraphs(self):
        """Test loading DOCX file with empty paragraphs."""
        mock_file = Mock()
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            mock_doc = Mock()
            mock_para = Mock()
            mock_para.text = ""
            mock_doc.paragraphs = [mock_para]
            mock_docx.return_value = mock_doc
            
            with pytest.raises(ValueError, match="No readable text found"):
                _load_docx(mock_file)
    
    def test_docx_file_with_very_little_text(self):
        """Test loading DOCX file with very little text."""
        mock_file = Mock()
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            mock_doc = Mock()
            mock_para = Mock()
            mock_para.text = "Short"
            mock_doc.paragraphs = [mock_para]
            mock_docx.return_value = mock_doc
            
            with pytest.raises(ValueError, match="DOCX contains very little text"):
                _load_docx(mock_file)
    
    def test_corrupted_docx_file(self):
        """Test handling of corrupted DOCX file."""
        mock_file = Mock()
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            mock_docx.side_effect = Exception("DOCX is corrupted")
            
            with pytest.raises(ValueError, match="DOCX file appears to be corrupted"):
                _load_docx(mock_file)
    
    def test_general_docx_error(self):
        """Test handling of general DOCX errors."""
        mock_file = Mock()
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            mock_docx.side_effect = Exception("General DOCX error")
            
            with pytest.raises(Exception, match="Error reading DOCX file"):
                _load_docx(mock_file)


class TestIngestIntegration:
    """Integration tests for document ingestion functions."""
    
    def test_end_to_end_pdf_processing(self):
        """Test complete end-to-end PDF processing."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = SAMPLE_PDF_CONTENT
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader
            
            result = load_document(mock_file)
            
            assert result == SAMPLE_PDF_CONTENT
    
    def test_end_to_end_txt_processing(self):
        """Test complete end-to-end TXT processing."""
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.size = 1024
        mock_file.read.return_value = SAMPLE_TXT_CONTENT.encode('utf-8')
        
        result = load_document(mock_file)
        
        assert result == SAMPLE_TXT_CONTENT
    
    def test_end_to_end_docx_processing(self):
        """Test complete end-to-end DOCX processing."""
        mock_file = Mock()
        mock_file.name = "test.docx"
        mock_file.size = 1024
        
        with patch('utils.ingest.docx.Document') as mock_docx:
            mock_doc = Mock()
            mock_para = Mock()
            mock_para.text = SAMPLE_DOCX_CONTENT
            mock_doc.paragraphs = [mock_para]
            mock_docx.return_value = mock_doc
            
            result = load_document(mock_file)
            
            assert result == SAMPLE_DOCX_CONTENT
    
    def test_file_size_validation(self):
        """Test file size validation across different file types."""
        file_types = ["test.pdf", "test.txt", "test.docx"]
        
        for file_type in file_types:
            mock_file = Mock()
            mock_file.name = file_type
            mock_file.size = 15 * 1024 * 1024  # 15MB (too large)
            
            with pytest.raises(ValueError, match="File too large"):
                load_document(mock_file)
    
    def test_error_handling_consistency(self):
        """Test consistent error handling across file types."""
        file_types = ["test.pdf", "test.txt", "test.docx"]
        
        for file_type in file_types:
            mock_file = Mock()
            mock_file.name = file_type
            mock_file.size = 1024
            
            # Test with file loading errors
            if file_type.endswith('.pdf'):
                with patch('utils.ingest._load_pdf') as mock_load:
                    mock_load.side_effect = Exception("Loading failed")
                    with pytest.raises(Exception, match="Failed to load document"):
                        load_document(mock_file)
            elif file_type.endswith('.txt'):
                with patch('utils.ingest._load_txt') as mock_load:
                    mock_load.side_effect = Exception("Loading failed")
                    with pytest.raises(Exception, match="Failed to load document"):
                        load_document(mock_file)
            elif file_type.endswith('.docx'):
                with patch('utils.ingest._load_docx') as mock_load:
                    mock_load.side_effect = Exception("Loading failed")
                    with pytest.raises(Exception, match="Failed to load document"):
                        load_document(mock_file)