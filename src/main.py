import os
from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Set up API keys
os.environ["SERPER_API_KEY"] = "111" # serper.dev API key

app = FastAPI()

search_tool = SerperDevTool()

llm_deepseek = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://127.0.0.1:11434"
)

# Define your agents with roles and goals
researcher = Agent(
  role='Researcher',
  goal="Find and summarize information about specific {topic}. If necessary, use the search tool to find relevant information.",
  backstory="You are an experienced researcher with attention to detail",
  tools=[search_tool],
  verbose=True,
  allow_delegation=False,
  llm=llm_deepseek
)
writer = Agent(
  role='Writer',
  goal='Write a compelling article about {topic}',
  backstory="You're a criative and famous technical writer, specialized on writing data related content",
  verbose=True,
  allow_delegation=False,
  llm=llm_deepseek
)

# Definir o modelo de entrada
class BlogRequirements(BaseModel):
    topic: str

# Definir a rota para executar a tarefa
@app.post("/blog")
async def blog(inputs: BlogRequirements):

    # Create tasks for your agents
    task1 = Task(
        description="""
            Research the {topic} and gather key points.
            Make sure you find any interesting and relevant information 
        """,
        expected_output="""
            A list with 10 bullet points of the most relevant information about {topic}
        """,
        agent=researcher
    )

    task2 = Task(
        description="""
            Write a compelling article about {topic} based on the research.
            Make it engaging and informative.
        """,
        expected_output="""
            A blog post with at least 500 words.""",
        agent=writer
    )

    # Instantiate your crew with a sequential process - TWO AGENTS!
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True,
        process=Process.sequential
    )
    result = crew.kickoff(inputs={'topic': inputs.topic})
    return {"result": result}

# Rodar o servidor usando Uvicorn
if __name__ == "__main__":
    import uvicorn
    print(">>>>>>>>>>>> version V0.0.1")
    uvicorn.run(app, host="0.0.0.0", port=8000)