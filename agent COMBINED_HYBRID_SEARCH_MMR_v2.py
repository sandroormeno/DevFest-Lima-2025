from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os

from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai import Agent, RunContext
from graphiti_core import Graphiti
from pydantic_ai import Agent, ModelSettings


from graphiti_core.llm_client.groq_client import GroqClient, LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# NOde de busqueda
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_MMR

load_dotenv()

# ========== Define dependencies ==========
@dataclass
class GraphitiDependencies:
    """Dependencies for the Graphiti agent."""
    graphiti_client: Graphiti

# ========== Helper function to get model configuration ==========
def get_model():
    """Configure and return the LLM model to use."""
    model_choice = os.getenv('MODEL_CHOICE')
    api_key = os.getenv('GROQ_API_KEY', 'value does not exist')

    return GroqModel(model_choice, provider=GroqProvider(api_key=api_key))

# ========== Create the Graphiti agent ==========
graphiti_agent = Agent(
    get_model(),
    system_prompt="""Eres un asistente útil con acceso a un grafo de conocimiento repleto de datos temporales sobre LLM. 
    Cuando el usuario te haga una pregunta, utiliza tu herramienta de búsqueda para consultar el grafo de conocimiento 
    y responde con sinceridad. Admite con franqueza cuando no encuentres la información necesaria para responder a la pregunta. 
    y responde de manera simple y con pocas palabras""",
    model_settings={'temperature': 0.0},
    deps_type=GraphitiDependencies
)

# ========== Define a result model for Graphiti search ==========
class GraphitiSearchResult(BaseModel):
    """Model representing a search result from Graphiti."""
    uuid: str = Field(description="The unique identifier for this fact")
    fact: str = Field(description="The factual statement retrieved from the knowledge graph")
    valid_at: Optional[str] = Field(None, description="When this fact became valid (if known)")
    invalid_at: Optional[str] = Field(None, description="When this fact became invalid (if known)")
    source_node_uuid: Optional[str] = Field(None, description="UUID of the source node")

# ========== Graphiti search tool ==========
@graphiti_agent.tool
async def search_graphiti(ctx: RunContext[GraphitiDependencies], query: str) -> List[GraphitiSearchResult]:
    """Search the Graphiti knowledge graph with the given query.
    
    Args:
        ctx: The run context containing dependencies
        query: The search query to find information in the knowledge graph
        
    Returns:
        A list of search results containing facts that match the query
    """
    # Access the Graphiti client from dependencies
    graphiti = ctx.deps.graphiti_client
    
    try:
        # Perform the search
        #results = await graphiti.search(query)
        node_search_config = COMBINED_HYBRID_SEARCH_MMR.model_copy(deep=True)
        node_search_config.limit = 5# Limit to 5 results

        results = await graphiti._search(query, config=node_search_config,)

        formatted_results = []
        # -----------------
        # results for nodes
        # -----------------
        for result in results.nodes:
            node_summary = result.summary[:100] + '...' if len(result.summary) > 100 else result.summary
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=node_summary,
                #source_node_uuid=result.source_node_uuid if hasattr(result, 'source_node_uuid') else None
            )            
            
            formatted_results.append(formatted_result)
        # -----------------
        # results for edges
        # -----------------
        for edge in results.edges:
    
            formatted_result = GraphitiSearchResult(
                uuid=edge.uuid,
                fact=edge.fact,
                #source_node_uuid=edge.source_node_uuid if hasattr(edge, 'source_node_uuid') else None
            )
            formatted_results.append(formatted_result)
        
        return formatted_results
    except Exception as e:
        # Log the error but don't close the connection since it's managed by the dependency
        print(f"Error searching Graphiti: {str(e)}")
        raise

# ========== Main execution function ==========
async def main():
    """Run the Graphiti agent with user queries."""
    print("Graphiti Agent - COMBINED_HYBRID_SEARCH_MMR")
    print("Enter 'exit' to quit the program.")
    
    # Initialize Graphiti with Neo4j connection
    llm_config=LLMConfig(
        api_key=os.getenv('GROQ_API_KEY', 'value does not exist'),
        model="groq:llama-3.1-8b-instant",
        small_model="groq:llama-3.1-8b-instant"
    )

    llm_client = GroqClient(config=llm_config)

    graphiti_client = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "123456789", 
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="abc",
                embedding_model="mxbai-embed-large",
                embedding_dim=512,
                base_url="http://localhost:11434/v1",
            )
        ),
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )
  
    # Initialize the graph database with graphiti's indices if needed
    try:
        await graphiti_client.build_indices_and_constraints()
        print("Graphiti indices built successfully.")
    except Exception as e:
        print(f"Note: {str(e)}")
        print("Continuing with existing indices...")

    console = Console()
    messages = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    # Pass the Graphiti client as a dependency
                    deps = GraphitiDependencies(graphiti_client=graphiti_client)
                    
                    async with graphiti_agent.run_stream(
                        user_input, message_history=messages, deps=deps
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        # Close the Graphiti connection when done
        await graphiti_client.close()
        print("\nGraphiti connection closed.") 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise
