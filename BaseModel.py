from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    name: str
    age: int
    studying: bool
    
student_user = User("1dw23", 28, True)