from pydantic import BaseModel, Field

class SearchStatementAIResponse(BaseModel):
    """
    Structured output for search statment generation from user's question.
    """

    statement : str = Field(..., description="The actual search statement generated from the user's question using the file summary provided as context")
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="0.0 to 1.0 score indicating certainty in using the statement as answer to the question or just a statement that can be used for better search retrieval."
    )
