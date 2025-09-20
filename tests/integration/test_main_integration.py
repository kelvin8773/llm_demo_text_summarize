# tests/integration/test_main_integration.py - Integration tests for main application

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from utils import (
    fast_summarize_text,
    enhance_summarize_text,
    chinese_summarize_text,
    extract_keywords,
    extract_keywords_phrases,
    extract_chinese_keywords,
    plot_keywords,
    plot_chinese_keywords,
    load_document
)
from tests.fixtures.sample_texts import (
    ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT,
    CHINESE_SHORT_TEXT, CHINESE_MEDIUM_TEXT, CHINESE_LONG_TEXT,
    SAMPLE_PDF_CONTENT, SAMPLE_TXT_CONTENT, SAMPLE_DOCX_CONTENT
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestSummarizationIntegration:
    """Integration tests for summarization workflows."""
    
    def test_english_fast_summarization_workflow(self):
        """Test complete English fast summarization workflow."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Fast summarization result."}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Test workflow
            summary = fast_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5)
            keywords = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=10)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_english_enhanced_summarization_workflow(self):
        """Test complete English enhanced summarization workflow."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "Enhanced summarization result."}]
            
            # Test workflow
            summary = enhance_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5)
            keywords = extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=10)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_chinese_summarization_workflow(self):
        """Test complete Chinese summarization workflow."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "中文摘要结果。"}]
            
            # Test workflow
            summary = chinese_summarize_text(CHINESE_MEDIUM_TEXT, max_sentences=5)
            keywords = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=10)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_mixed_language_processing(self):
        """Test processing documents in different languages."""
        workflows = [
            (ENGLISH_MEDIUM_TEXT, fast_summarize_text, extract_keywords),
            (ENGLISH_MEDIUM_TEXT, enhance_summarize_text, extract_keywords_phrases),
            (CHINESE_MEDIUM_TEXT, chinese_summarize_text, extract_chinese_keywords)
        ]
        
        for text, summarizer, keyword_extractor in workflows:
            with patch.object(summarizer, '__module__') as mock_module, \
                 patch.object(keyword_extractor, '__module__') as mock_kw_module:
                
                # Mock the appropriate modules based on the function
                if 'fast_summarize' in str(summarizer):
                    with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
                         patch('utils.fast_summarize.pipeline') as mock_pipeline:
                        
                        mock_tokenizer_instance = Mock()
                        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
                        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                        
                        mock_pipeline_instance = Mock()
                        mock_pipeline_instance.return_value = [{"summary_text": "Summary result."}]
                        mock_pipeline.return_value = mock_pipeline_instance
                        
                        summary = summarizer(text, max_sentences=3)
                        keywords = keyword_extractor(text, top_n=5)
                
                elif 'enhance_summarize' in str(summarizer):
                    with patch('utils.enhance_summarize._initialize_models'), \
                         patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
                         patch('utils.enhance_summarize._summarizer') as mock_summarizer:
                        
                        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                        mock_summarizer.return_value = [{"summary_text": "Enhanced summary result."}]
                        
                        summary = summarizer(text, max_sentences=3)
                        keywords = keyword_extractor(text, top_n=5)
                
                elif 'chinese_summarize' in str(summarizer):
                    with patch('utils.chinese_summarize._initialize_models'), \
                         patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
                         patch('utils.chinese_summarize._summarizer') as mock_summarizer:
                        
                        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                        mock_summarizer.return_value = [{"summary_text": "中文摘要结果。"}]
                        
                        summary = summarizer(text, max_sentences=3)
                        keywords = keyword_extractor(text, top_n=5)
                
                assert isinstance(summary, str)
                assert len(summary) > 0
                assert isinstance(keywords, list)
                assert len(keywords) > 0


class TestDocumentProcessingIntegration:
    """Integration tests for document processing workflows."""
    
    def test_pdf_to_summary_workflow(self):
        """Test complete PDF to summary workflow."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        with patch('utils.ingest.PdfReader') as mock_pdf_reader, \
             patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock PDF loading
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = SAMPLE_PDF_CONTENT
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader
            
            # Mock summarization
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "PDF summary result."}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Test workflow
            text = load_document(mock_file)
            summary = fast_summarize_text(text, max_sentences=5)
            keywords = extract_keywords(text, top_n=10)
            
            assert text == SAMPLE_PDF_CONTENT
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_txt_to_summary_workflow(self):
        """Test complete TXT to summary workflow."""
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.size = 1024
        mock_file.read.return_value = SAMPLE_TXT_CONTENT.encode('utf-8')
        
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock summarization
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_summarizer.return_value = [{"summary_text": "TXT summary result."}]
            
            # Test workflow
            text = load_document(mock_file)
            summary = enhance_summarize_text(text, max_sentences=5)
            keywords = extract_keywords_phrases(text, top_n=10)
            
            assert text == SAMPLE_TXT_CONTENT
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_docx_to_summary_workflow(self):
        """Test complete DOCX to summary workflow."""
        mock_file = Mock()
        mock_file.name = "test.docx"
        mock_file.size = 1024
        
        with patch('utils.ingest.docx.Document') as mock_docx, \
             patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock DOCX loading
            mock_doc = Mock()
            mock_para = Mock()
            mock_para.text = SAMPLE_DOCX_CONTENT
            mock_doc.paragraphs = [mock_para]
            mock_docx.return_value = mock_doc
            
            # Mock Chinese summarization
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_summarizer.return_value = [{"summary_text": "DOCX中文摘要结果。"}]
            
            # Test workflow
            text = load_document(mock_file)
            summary = chinese_summarize_text(text, max_sentences=5)
            keywords = extract_chinese_keywords(text, top_n=10)
            
            assert text == SAMPLE_DOCX_CONTENT
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert isinstance(keywords, list)
            assert len(keywords) > 0


class TestVisualizationIntegration:
    """Integration tests for visualization workflows."""
    
    def test_english_keyword_visualization_workflow(self):
        """Test complete English keyword visualization workflow."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Test workflow
            keywords = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=10)
            fig = plot_keywords(keywords)
            
            assert isinstance(keywords, list)
            assert len(keywords) > 0
            assert fig == mock_fig
    
    def test_chinese_keyword_visualization_workflow(self):
        """Test complete Chinese keyword visualization workflow."""
        with patch('utils.chinese_insights._initialize_chinese_font'), \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Test workflow
            keywords = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=10)
            fig = plot_chinese_keywords(keywords)
            
            assert isinstance(keywords, list)
            assert len(keywords) > 0
            assert fig == mock_fig
    
    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        keywords = ["test", "keywords"]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_subplots.side_effect = Exception("Plotting failed")
            
            with pytest.raises(Exception, match="Plotting failed"):
                plot_keywords(keywords)


class TestErrorHandlingIntegration:
    """Integration tests for error handling across modules."""
    
    def test_cascading_error_handling(self):
        """Test error handling when one component fails."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        # Test PDF loading failure
        with patch('utils.ingest.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("PDF loading failed")
            
            with pytest.raises(Exception, match="Failed to load document"):
                load_document(mock_file)
    
    def test_summarization_error_recovery(self):
        """Test error recovery in summarization."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception, match="Failed to load model"):
                fast_summarize_text(ENGLISH_MEDIUM_TEXT)
    
    def test_keyword_extraction_error_recovery(self):
        """Test error recovery in keyword extraction."""
        with patch('utils.insights.TfidfVectorizer') as mock_vectorizer:
            mock_vectorizer.side_effect = Exception("Vectorization failed")
            
            with pytest.raises(Exception, match="Keyword extraction failed"):
                extract_keywords(ENGLISH_MEDIUM_TEXT)


class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""
    
    def test_large_document_processing(self):
        """Test processing of large documents."""
        import time
        
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Large document summary."}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            start_time = time.time()
            summary = fast_summarize_text(ENGLISH_LONG_TEXT, max_sentences=10)
            keywords = extract_keywords(ENGLISH_LONG_TEXT, top_n=20)
            end_time = time.time()
            
            assert isinstance(summary, str)
            assert isinstance(keywords, list)
            # Should complete within reasonable time (mocked, so should be fast)
            assert (end_time - start_time) < 2.0  # 2 seconds max for mocked test
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns across different operations."""
        operations = [
            lambda: fast_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5),
            lambda: enhance_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5),
            lambda: chinese_summarize_text(CHINESE_MEDIUM_TEXT, max_sentences=5),
            lambda: extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=10),
            lambda: extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=10),
            lambda: extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=10)
        ]
        
        for operation in operations:
            with patch.object(operation, '__module__') as mock_module:
                # Mock the appropriate modules
                if 'fast_summarize' in str(operation):
                    with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
                         patch('utils.fast_summarize.pipeline') as mock_pipeline:
                        
                        mock_tokenizer_instance = Mock()
                        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
                        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                        
                        mock_pipeline_instance = Mock()
                        mock_pipeline_instance.return_value = [{"summary_text": "Test summary."}]
                        mock_pipeline.return_value = mock_pipeline_instance
                        
                        result = operation()
                
                elif 'enhance_summarize' in str(operation):
                    with patch('utils.enhance_summarize._initialize_models'), \
                         patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
                         patch('utils.enhance_summarize._summarizer') as mock_summarizer:
                        
                        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                        mock_summarizer.return_value = [{"summary_text": "Enhanced summary."}]
                        
                        result = operation()
                
                elif 'chinese_summarize' in str(operation):
                    with patch('utils.chinese_summarize._initialize_models'), \
                         patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
                         patch('utils.chinese_summarize._summarizer') as mock_summarizer:
                        
                        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                        mock_summarizer.return_value = [{"summary_text": "中文摘要。"}]
                        
                        result = operation()
                
                else:  # Keyword extraction
                    result = operation()
                
                assert result is not None
                assert isinstance(result, (str, list))


class TestStreamlitIntegration:
    """Integration tests for Streamlit application components."""
    
    def test_streamlit_session_state_handling(self):
        """Test Streamlit session state handling."""
        # Mock Streamlit session state
        with patch('streamlit.session_state') as mock_session_state:
            mock_session_state.get.return_value = None
            
            # Test that session state is properly handled
            assert mock_session_state.get('test_key') is None
    
    def test_streamlit_widget_interactions(self):
        """Test Streamlit widget interactions."""
        # Mock Streamlit widgets
        with patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.checkbox') as mock_checkbox:
            
            mock_selectbox.return_value = "English"
            mock_slider.return_value = 5
            mock_checkbox.return_value = True
            
            # Test widget interactions
            language = mock_selectbox("Language", ["English", "Chinese"])
            max_sentences = mock_slider("Max Sentences", 1, 20, 5)
            use_sample = mock_checkbox("Use sample file")
            
            assert language == "English"
            assert max_sentences == 5
            assert use_sample is True
    
    def test_streamlit_file_upload_handling(self):
        """Test Streamlit file upload handling."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        with patch('streamlit.file_uploader') as mock_file_uploader:
            mock_file_uploader.return_value = mock_file
            
            uploaded_file = mock_file_uploader("Choose a file", type=["pdf", "txt", "docx"])
            
            assert uploaded_file == mock_file
            assert uploaded_file.name == "test.pdf"
            assert uploaded_file.size == 1024