import os
from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

app = FastAPI()

search_tool = SerperDevTool()

llm_deepseek = LLM(
    model="ollama/deepseek-r1:7b",
    base_url="http://localhost:11434"
)

# Define your agents with roles and goals
researcher = Agent(
  role='Researcher',
  goal='Discover new insights',
  backstory="You're a world class researcher working on a major data science company",
  tools=[search_tool]
  verbose=True,
  allow_delegation=False,
  llm=llm_deepseek
)
writer = Agent(
  role='Writer',
  goal='Create engaging content',
  backstory="You're a famous technical writer, specialized on writing data related content",
  tools=[search_tool]
  verbose=True,
  allow_delegation=False,
  llm=llm_deepseek
)

# Definir o modelo de entrada
class BlogRequirements(BaseModel):
    job_requirements: str

# Definir a rota para executar a tarefa
@app.post("/blog")
async def blog(req: BlogRequirements):

    # Create tasks for your agents
    task1 = Task(
        description="""
            Investigate the latest AI trends
            Make sure you find any interesting and relevant information 
            Requirements: {req.job_requirements}
        """,
        expected_output="""
            A list with 10 bullet points of the most relevant information about AI Agents
        """,
        agent=researcher
    )

    task2 = Task(
        description="""
            Write a blog post on AI advancements
            Make sure you write a compelling article with the latest
        """,
        expected_output="""
            A blog post with at least 500 words on the latest AI advancements
        """,
        agent=writer
    )

    # Instantiate your crew with a sequential process - TWO AGENTS!
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True,
        llm=llm_deepseek, 
    )
    result = crew.kickoff(inputs={'blog_requirements': req.blog_requirements})
    return {"result": result}

# Rodar o servidor usando Uvicorn
if __name__ == "__main__":
    import uvicorn
    print(">>>>>>>>>>>> version V0.0.1")
    uvicorn.run(app, host="0.0.0.0", port=8000)