from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import openai 
import os 

load_dotenv()

# Set open ai api key
openai.apikey = os.getenv("OPENAI_API_KEY") # type:  ignore

chat = ChatOpenAI(model_name = "gpt-3.5-turbo-0613",temperature=0) # type:ignore

def add_or_update_user_function(user_data_dict, embedder, pinecone_index):
    """
    Function to add or update a user in the weavaite vector database.

    Args:
        user_data_dict (dict): A dictionary containing the user data.
        embedder (Embedder): An instance of the Embedder class.
        pinecone_client (WeaviateClient): An instance of the WeaviateClient class.
        class_name (str): The name of the weaviate class/index to add the user data to.

    Returns:
        bool: True if the user data was added successfully, False otherwise.
    """

    required_keys = ["name", "description", "rating", "portfolio", "keywords", 
                     "hourly_rate", "username", "profile_url", "provider_id", 
                     "profile_picture"]

    # validate if "description" present or ont or if its blank,
    #If any of the keys from required_keys not present in provider_data_obj, then add the key and value as ""
    for key in required_keys:
        if key not in user_data_dict:
            user_data_dict[key] = "";

    # raise error if "description" is balnk string like ""
    if user_data_dict["description"] == "":
        raise ValueError("description is blank")
    else:
        temp = {}
        temp["id"] = user_data_dict["provider_id"]

        # remove "provider_id" key from dict
        del user_data_dict["provider_id"]

        temp["metadata"] = user_data_dict
        temp["values"] = embedder.embed_documents(temp["metadata"]["description"])[0]

        try:
            # Load into weaviate
            upsert_response = pinecone_index.upsert(
                vectors=[temp],
                batch_size=100,
                show_progress=True
            )
            
            if upsert_response:
                return True
            else:
                return False
        except Exception as e:
            print("Error in adding data to weaviate", e)
            return False
    
def create_context(provider_dict):
    context = f"Name: {provider_dict['metadata']['name']}\n"
    context += f"Description: {provider_dict['metadata']['description']}\n"
    context += f"Rating: {provider_dict['metadata']['rating']}\n"
    context += f"Portfolio: {provider_dict['metadata']['portfolio']}\n"
    context += f"Hourly Rate: {provider_dict['metadata']['hourly_rate']}\n"
    context += f"Profile URL: {provider_dict['metadata']['profile_url']}"

    return context

def generate_sales_pitch(client_query, context):
    """
    Generates a sales pitch for a candidate based on the client query and candidate details.

    Parameters:
        query (str): The client query.
        context (str): The candidate details.

    Returns:
        str: The professional sales pitch for the candidate, limited to 50 words.
    """

    # Create a chat prompt template with the client query and candidate details.
    # The sales pitch should include all about the candidate.
    # sales_pitch:
    human_template = """Client query: {_query}

    Candidate details:
    {context}

    Create a professional sales pitch showing why this candidate is good for this requirement with in 50 words. 
    The sales pitch should include all about the candidate.

    sales_pitch:
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [human_message_prompt]
    )

    # get a chat completion from the formatted messages
    response = chat(
        chat_prompt.format_prompt(
            context=context, _query = client_query
        ).to_messages()
    )

    return response.content

def find_recommended_providers_function(_query, embedder, pinecone_index):
    """
    Find recommended providers based on a query and return a list of matching providers.
    
    Args:
        query (str): The query string used to generate the embedding.
        embedder (Embedder): The embedding model used to generate embeddings.
        pinecone_client (WeaviateClient): The client object used to interact with Weaviate.
        class_name (str): The name of the class in Weaviate.
        
    Returns:
        list: A list of dictionaries, each containing information about a recommended provider.
    """

    # Generate embedding of query
    query_embedding = embedder.embed_query(_query)

    query_response = pinecone_index.query(
            top_k=3,
            include_values=False,
            include_metadata=True,
            vector=query_embedding,
        )
    
    # Convert from pinecone 
    query_response = query_response.to_dict()

    final_providerx = []

    if len(query_response["matches"]) > 0:
        for provider in query_response["matches"]:


            # create context
            context = create_context(provider)
            # generate sales pitch
            sales_pitch = generate_sales_pitch(_query, context)
            # Add sales pitch generated by GPT3.5
            provider["sales_pitch"] = sales_pitch

            final_providerx.append(provider)
    
    return {"recommended_providers" : final_providerx}


