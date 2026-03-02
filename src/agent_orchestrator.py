import autogen
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

config_list = [{"model": "gpt-4-turbo", "api_key": os.getenv("OPENAI_API_KEY")}]

architect_agent = autogen.AssistantAgent(
    name="System_Architect",
    llm_config={"config_list": config_list},
    system_message="You are a Principal Systems Architect. Design highly scalable, distributed microservices architectures. Output strict API contracts and database schemas."
)

ml_engineer_agent = autogen.AssistantAgent(
    name="Senior_ML_Engineer",
    llm_config={"config_list": config_list},
    system_message="You are a Senior ML Engineer. Write robust, production-ready Python/FastAPI code and Dockerfiles based on the Architect's design. Optimize for latency."
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={"work_dir": "../generated_infrastructure", "use_docker": False}
)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found. Please check your .env file.")
        exit(1)
        
    user_proxy.initiate_chat(
        architect_agent,
        message="Design and implement a low-latency, distributed RAG pipeline capable of vectorizing and querying millions of academic papers."
    )