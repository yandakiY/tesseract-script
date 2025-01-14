from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel , Field
import uvicorn as uvicorn

app = FastAPI()

class Person(BaseModel):
    name: str = Field(... ,title="Name of the person", description="Name of the person", min_length=2 ,max_length=100)
    age: int = Field(... ,title="Age of the person", description="Age of the person", ge=1)
    description: Optional[str] = Field(None,min_length=2 ,max_length=100)
    
list_persons : list[Person] = []

@app.get("/hello")
def home():
    return {"message": "Hello, World!"}

@app.get("/about")
def about():
    return {"message": "This is a simple FastAPI application."}

@app.get("/test/{id}")
async def test(id : int):
    return {"message": f"This is a POST method. {id}"}

@app.post("/persons")
def create_person(person: Person):
    list_persons.append(person)
    return person

@app.get("/persons")
def list_person():
    return {"persons":list_persons}

def main():
    uvicorn.run(app, host="0.0.0.0", port=9191 , reload=True)
    
if __name__ == '__main__':
    main()