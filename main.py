#    Copyright 2025 Rashed Talukder
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import time
import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
UPLOADED_FILE_ID = os.getenv("UPLOADED_FILE_ID", "")


def upload_to_vector_store(aoai_client, file_path, vector_store_name):
    global UPLOADED_FILE_ID

    if not UPLOADED_FILE_ID:
        with open(file_path, "rb") as file_content:
            # Upload file and get the file ID
            uploaded_file = aoai_client.files.create(
                file=file_content,
                purpose="assistants"
            )
            UPLOADED_FILE_ID = uploaded_file.id
            print(f"File ID: {UPLOADED_FILE_ID}")
    else:
        print(f"Using existing file ID: {UPLOADED_FILE_ID}")

    # Create a vector store called "Financial Statements"
    vector_store = aoai_client.vector_stores.create(
        name=vector_store_name,
        expires_after={
            "anchor": "last_active_at",
            "days": 7
        })
    print(f"Vector Store ID: {vector_store.id}")

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    vector_store_file = aoai_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        chunking_strategy={
            "type": "static",
            "static": {
                "max_chunk_size_tokens": 100,
                "chunk_overlap_tokens": 20
            }
        },
        file_id=UPLOADED_FILE_ID,
        attributes={
            "source": "Contoso",
            "category": "Marketing",
        }
    )

    # Poll the status of the file batch until it is completed
    while True:
        file_uploaded = aoai_client.vector_stores.files.poll(
            file_id=vector_store_file.id,
            vector_store_id=vector_store.id,
        )
        if file_uploaded.status == "completed":
            print("File batch completed successfully.")
            break
        elif file_uploaded.status == "failed":
            print("File batch failed. %s", file_uploaded.last_error())
            cleanup(aoai_client, [vector_store.id], [])
            return
        else:
            print(f"File batch status: {file_uploaded.status}")
            time.sleep(5)

    return vector_store


def cleanup(aoai_client, vector_store_ids, response_ids):
    """
    Cleans up the vector stores and responses created during the test.
    """
    for vector_store_id in vector_store_ids:
        try:
            print(f"Deleting vector store {vector_store_id}...")
            aoai_client.vector_stores.delete(vector_store_id)
        except Exception as e:
            print(f"Error deleting vector store {vector_store_id}: {e}")

    for response_id in response_ids:
        try:
            print(f"Deleting response {response_id}...")
            aoai_client.responses.delete(response_id)
        except Exception as e:
            print(f"Error deleting response {response_id}: {e}")

    print("Deleting uploaded file...")
    aoai_client.files.delete(UPLOADED_FILE_ID)
    print("Cleanup completed.")


def main():
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=AZURE_OPENAI_API_VERSION
    )

    try:
        vector_store = upload_to_vector_store(
            client,
            file_path="./Contoso_Brochure.pdf",
            vector_store_name="Travel Brochure"
        )

        response = client.responses.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            input="What is Contoso Travel Agency's phone number?",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                    "max_num_results": 1,
                    "filters": {
                        "type": "eq",
                        "key": "category",
                        "value": "Marketing"
                    },
                    "ranking_options": {
                        "ranker": "auto",
                        "score_threshold": 0.01
                    },
                }
            ],
            # Includes search results in the response
            include=["file_search_call.results"]
        )

        print(response.model_dump_json(indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")

    cleanup(client, [vector_store.id], [response.id])


if __name__ == "__main__":
    main()
