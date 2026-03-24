# Cognitive Agent Syndicate

A multi-agent autonomous system designed to architect, generate, and document highly scalable microservices. This system utilizes customized LLM agents acting as a Principal Systems Architect and a Senior ML Engineer to translate high-level system requirements into production-ready infrastructure code.

### Core Architecture
* System Architect Agent: Ingests requirements, outputs strict API contracts, database schemas, and system design logic.
* Senior ML Engineer Agent: Ingests the Architect's specifications and generates robust, low-latency Python/FastAPI code.
* Execution Environment: Safely isolates generated code and Dockerfiles into a dedicated `generated_infrastructure/` directory.

### Tech Stack
* Language: Python 3.10+
* Frameworks: PyAutoGen, FastAPI
* AI/LLM: OpenAI
* Environment: Docker

### Build and Execution
Note: Requires Python 3.10+ and an OpenAI API Key.

1. Clone the repository
   ```bash
   git clone https://github.com/nolanefe/Cognitive-Agent-Syndicate.git
   cd Cognitive-Agent-Syndicate
   ```

2. Run and execute
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
