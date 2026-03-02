# Cognitive Agent Syndicate

A multi-agent autonomous system designed to architect, generate, and document highly scalable microservices. This system utilizes customized LLM agents acting as a Principal Systems Architect and a Senior ML Engineer to translate high-level system requirements into production-ready infrastructure code.

### System Architecture

1. **System Architect Agent:** Ingests requirements, outputs strict API contracts, database schemas, and system design logic.
2. **Senior ML Engineer Agent:** Ingests the Architect's specifications and generates robust, low-latency Python/FastAPI code.
3. **Execution Environment:** Safely isolates generated code and Dockerfiles into a dedicated `generated_infrastructure/` directory.

### Build and Execution

**1. Clone the repository**
```bash
git clone git clone https://github.com/nolanefe/Cognitive-Agent-Syndicate.git
cd Cognitive-Agent-Syndicate 