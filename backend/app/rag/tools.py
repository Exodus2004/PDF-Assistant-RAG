"""
Custom tools for the Agentic RAG system.
Defines PDF Search, Web Research, and Math tools.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from app.rag.retriever import retrieve
from app.rag.graph_retriever import get_entity_context

logger = logging.getLogger(__name__)

class PDFSearchSchema(BaseModel):
    query: str = Field(description="The search query to look for in the PDF documents.")

class PDFSearchTool(BaseTool):
    name: str = "pdf_search"
    description: str = (
        "Useful for searching and retrieving relevant information from uploaded PDF documents. "
        "Use this for any questions about the content of the documents. "
        "The tool will automatically search across all selected documents."
    )
    args_schema: Type[BaseModel] = PDFSearchSchema
    
    user_id: str
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    # We'll store sources here to retrieve them after agent execution
    last_sources: List[Dict[str, Any]] = []

    def _run(self, query: str) -> str:
        """Execute the search."""
        try:
            chunks = retrieve(
                query=query,
                user_id=self.user_id,
                document_id=self.document_id,
                document_ids=self.document_ids,
            )
            
            # Save for later retrieval
            self.last_sources = chunks
            
            if not chunks:
                return "No relevant information found in the documents."

            # Format chunks for the LLM
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(
                    f"Excerpt {i} ({chunk['filename']}, Page {chunk['page']}):\n{chunk['text']}"
                )
            
            # Also try to get GraphRAG context
            graph_context = get_entity_context(
                query=query,
                user_id=self.user_id,
                document_id=self.document_id,
            )
            
            main_context = "\n\n".join(context_parts)
            if graph_context:
                return f"{main_context}\n\nAdditional Relationships found:\n{graph_context}"
            
            return main_context
        except Exception as e:
            logger.error(f"PDFSearchTool error: {e}")
            return f"Error searching documents: {str(e)}"

class MathSchema(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate (e.g., '2 + 2' or 'Revenue - Expenses').")

class MathTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "Useful for performing mathematical calculations and evaluating numerical expressions. "
        "Use this when the user asks for sums, differences, or complex math based on document data."
    )
    args_schema: Type[BaseModel] = MathSchema

    def _run(self, expression: str) -> str:
        """Execute the math evaluation safely."""
        try:
            # Simple eval with some restrictions
            # In a production environment, use a more secure math parser
            import math
            allowed_names = {"__builtins__": None, "math": math}
            # Add common math functions
            for name in ["sum", "min", "max", "abs", "round"]:
                allowed_names[name] = __builtins__.get(name)
            
            # Remove any potentially dangerous characters
            clean_expression = "".join(c for c in expression if c in "0123456789+-*/()., ")
            
            result = eval(clean_expression, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}. Please ensure it's a valid numerical expression."

# Placeholder for Web Search Tool (to be implemented in Issue #220)
class WebSearchSchema(BaseModel):
    query: str = Field(description="The query to search the live web for.")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Useful for fact-checking information or finding live data from the internet. "
        "Use this only when the PDF content is insufficient or outdated."
    )
    args_schema: Type[BaseModel] = WebSearchSchema

    def _run(self, query: str) -> str:
        return "Web search is currently in maintenance mode. Please rely on document content."
