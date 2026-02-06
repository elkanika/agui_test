import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import os
import sys

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AsistenteFisica

@pytest.fixture
def mock_dependencies():
    with patch('agent.ChatGoogleGenerativeAI') as mock_llm, \
         patch('agent.AsyncQdrantClient') as mock_qdrant, \
         patch('agent.AutoTokenizer') as mock_tokenizer, \
         patch('agent.AutoModel') as mock_model, \
         patch('agent.PdfReader') as mock_pdf_reader, \
         patch('agent.Runner') as mock_runner, \
         patch('agent.LlmAgent') as mock_llm_agent:

        # Setup common mocks
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        mock_qdrant_instance = AsyncMock()
        mock_qdrant.return_value = mock_qdrant_instance

        yield {
            'llm': mock_llm,
            'llm_instance': mock_llm_instance,
            'qdrant': mock_qdrant,
            'qdrant_instance': mock_qdrant_instance,
            'tokenizer': mock_tokenizer,
            'model': mock_model,
            'pdf_reader': mock_pdf_reader,
            'runner': mock_runner,
            'llm_agent': mock_llm_agent
        }

@pytest.fixture
def asistente(mock_dependencies):
    # Prevent __init__ from running full setup if we want granular control,
    # but here we want to test init too, so we let it run with mocks.
    # We might need to mock environment variables
    with patch.dict(os.environ, {
        "GOOGLE_API_KEY": "fake_key",
        "QDRANT_URL": "http://fake:6333",
        "QDRANT_KEY": "fake_qdrant_key"
    }):
        app = AsistenteFisica()
        # Initialize components since __init__ doesn't do it
        app.inicializar_componentes()
        return app

def test_initialization(asistente, mock_dependencies):
    """Test that the assistant initializes correctly."""
    assert asistente.llm is not None
    assert asistente.memoria_semantica is not None
    assert asistente.agents is not None
    assert 'classifier' in asistente.agents
    assert 'search' in asistente.agents
    assert 'response' in asistente.agents

    # Check that LLM was initialized with config
    mock_dependencies['llm'].assert_called()

def test_procesar_pdfs_temario(asistente, mock_dependencies):
    """Test PDF processing and syllabus extraction."""
    mock_pdf_reader = mock_dependencies['pdf_reader']
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock(), MagicMock()]
    mock_pdf.pages[0].extract_text.return_value = "Page 1 content. "
    mock_pdf.pages[1].extract_text.return_value = "Page 2 content."
    mock_pdf_reader.return_value = mock_pdf

    # Mock LLM response for syllabus
    mock_msg = MagicMock()
    mock_msg.content = "TEMA: 1. Kinematics\nSUBTEMAS: [Velocity, Acceleration]"
    mock_dependencies['llm_instance'].invoke.return_value = mock_msg

    with patch('os.path.exists', return_value=True):
        temario = asistente.procesar_pdfs_temario(["fake.pdf"])

    assert temario == "TEMA: 1. Kinematics\nSUBTEMAS: [Velocity, Acceleration]"
    assert asistente.contenido_completo == "\n--- Contenido de fake.pdf ---\nPage 1 content. Page 2 content."

    # Verify LLM was called with correct context
    args, _ = mock_dependencies['llm_instance'].invoke.call_args
    assert "Page 1 content" in args[0][0].content

@pytest.mark.asyncio
async def test_iniciar_flujo(asistente, mock_dependencies):
    """Test the main RAG flow."""
    # Mock specific agent responses
    # We need to mock _get_agent_response since it uses internal Runner logic that might be complex to fully mock via dependencies

    # However, let's try to mock the Runner execution if possible, or simpler: mock _get_agent_response directly on the instance

    with patch.object(asistente, '_get_agent_response', new_callable=AsyncMock) as mock_get_response:
        # Define side effects for the sequence of calls: [classifier, search, responder]
        mock_get_response.side_effect = [
            "TEMA: 1. Kinematics", # Classifier
            "kinematics velocity", # Search
            "Respuesta final sobre cinematica" # Responder
        ]

        # Mock search_documents
        with patch.object(asistente, 'search_documents', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"pdf": "doc1.pdf", "texto": "info about velocity", "similitud": 0.9}
            ]

            # Run flow
            respuesta = await asistente.iniciar_flujo("Que es la velocidad?")

            assert respuesta == "Respuesta final sobre cinematica"

            # Verify calls
            assert mock_get_response.call_count == 3
            # Check classifier call
            assert mock_get_response.call_args_list[0][0][0] == asistente.classifier_agent
            # Check search call
            assert mock_get_response.call_args_list[1][0][0] == asistente.search_agent
            # Check responder call
            assert mock_get_response.call_args_list[2][0][0] == asistente.response_agent

            mock_search.assert_called_with("kinematics velocity")

@pytest.mark.asyncio
async def test_iniciar_flujo_fallback(asistente):
    """Test fallback when an error occurs in the flow."""
    with patch.object(asistente, '_get_agent_response', side_effect=Exception("Explosion")):
        respuesta = await asistente.iniciar_flujo("Consulta explosiva")
        assert "error técnico" in respuesta
