from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    username: str = Field(description="The username of the user")
    password: str = Field(description="The password of the user")

class CreateSessionRequest(BaseModel):
    doc_id: int = Field(description="The id of the document the user wishes to interact with")
