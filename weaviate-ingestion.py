import weaviate
import json
import os
from weaviate.util import generate_uuid5
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

print(WEAVIATE_URL)
print(WEAVIATE_API_KEY)
# print(HF_API_KEY)

# --- Define Class Names ---
TABLE_CLASS_NAME = "PwcTable"
COLUMN_CLASS_NAME = "PwcColumn"

def main():
    # --- Initialize client ---
    try:
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
            additional_headers={"X-HuggingFace-Api-Key": HF_API_KEY}
        )
        print("✅ Connected to Weaviate.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # --- Do NOT delete existing schemas ---
    # Instead, create schemas only if they do not exist
    if not client.schema.exists(TABLE_CLASS_NAME):
        table_schema = {
            "class": TABLE_CLASS_NAME,
            "description": "High-level information about a data table.",
            "properties": [
                {"name": "tableName", "dataType": ["text"], "description": "The name of the table."},
                {"name": "tableDescription", "dataType": ["text"], "description": "The detailed description of the table."},
                {"name": "tableSummary", "dataType": ["text"], "description": "The summary of the table's purpose."},
                {"name": "fullMetadata", "dataType": ["text"], "description": "The complete original JSON metadata for the table."}
            ]
        }
        try:
            client.schema.create_class(table_schema)
            print(f"✅ Class '{TABLE_CLASS_NAME}' created.")
        except Exception as e:
            print(f"❌ Failed to create schema '{TABLE_CLASS_NAME}': {e}")
            return
    else:
        print(f"ℹ️ Class '{TABLE_CLASS_NAME}' already exists. Will append new entries.")

    if not client.schema.exists(COLUMN_CLASS_NAME):
        column_schema = {
            "class": COLUMN_CLASS_NAME,
            "description": "Detailed information about a single column within a table.",
            "vectorizer": "text2vec-huggingface",
            "moduleConfig": {
                "text2vec-huggingface": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "options": {"waitForModel": True},
                    "vectorizePropertyName": "content"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "A rich, contextual string combining table and column info for vectorization."
                },
                {"name": "tableName", "dataType": ["text"], "description": "Name of the parent table."},
                {"name": "columnName", "dataType": ["text"], "description": "Name of the column."},
                {"name": "columnDescription", "dataType": ["text"], "description": "Description of the column's purpose."},
                {
                    "name": "fromTable",
                    "dataType": [TABLE_CLASS_NAME],
                    "description": "A link to the parent table object."
                }
            ]
        }
        try:
            client.schema.create_class(column_schema)
            print(f"✅ Class '{COLUMN_CLASS_NAME}' created.")
        except Exception as e:
            print(f"❌ Failed to create schema '{COLUMN_CLASS_NAME}': {e}")
            return
    else:
        print(f"ℹ️ Class '{COLUMN_CLASS_NAME}' already exists. Will append new entries.")

    # --- Load data ---
    try:
        with open('ddl_json.json', 'r') as f:
            data = json.load(f)
        print(f"📄 Loaded {len(data)} tables from 'ddl_json.json'.")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # --- Error checker for batch results ---
    def check_batch_results(results: dict):
        if results:
            for result in results:
                if "result" in result and "errors" in result["result"]:
                    if "error" in result["result"]["errors"]:
                        print(f"❌ Batch error: {result['result']['errors']['error']}")

    # --- Get existing UUIDs to avoid duplicates ---
    def get_existing_uuids(class_name):
        uuids = set()
        try:
            # Query all objects, get their _additional { id }
            resp = client.query.get(class_name, ["_additional { id }"]).with_limit(10000).do()
            objs = resp.get("data", {}).get("Get", {}).get(class_name, [])
            for obj in objs:
                if "_additional" in obj and "id" in obj["_additional"]:
                    uuids.add(obj["_additional"]["id"])
        except Exception as e:
            print(f"⚠️ Could not fetch existing UUIDs for {class_name}: {e}")
        return uuids

    print("🔎 Fetching existing table and column UUIDs to avoid duplicates...")
    existing_table_uuids = get_existing_uuids(TABLE_CLASS_NAME)
    existing_column_uuids = get_existing_uuids(COLUMN_CLASS_NAME)

    # --- Start batch import ---
    print("🚀 Starting granular batch import (append mode)...")
    client.batch.configure(batch_size=50, callback=check_batch_results)

    with client.batch as batch:
        for table_data in data:
            table_name = table_data.get("table_name")
            table_description = table_data.get("description", "")
            table_summary = table_data.get("summary", "")
            
            if not table_name: continue

            table_uuid = generate_uuid5(table_name)
            if table_uuid not in existing_table_uuids:
                batch.add_data_object(
                    data_object={
                        "tableName": table_name,
                        "tableDescription": table_description,
                        "tableSummary": table_summary,
                        "fullMetadata": json.dumps(table_data)
                    },
                    class_name=TABLE_CLASS_NAME,
                    uuid=table_uuid
                )
                print(f"🔹 Queued Table: {table_name}")
            else:
                print(f"🔹 Table '{table_name}' already exists, skipping.")

            for column_data in table_data.get("columns", []):
                column_name = column_data.get("name")
                column_description = column_data.get("description", "")
                
                if not column_name: continue

                content_to_vectorize = (
                    f"Table: {table_name}. Column: {column_name}. "
                    f"Description: {column_description}. "
                    f"Table Purpose: {table_summary}"
                )

                column_object = {
                    "content": content_to_vectorize,
                    "tableName": table_name,
                    "columnName": column_name,
                    "columnDescription": column_description
                }
                
                column_uuid = generate_uuid5(f"{table_name}_{column_name}")
                if column_uuid not in existing_column_uuids:
                    batch.add_data_object(
                        data_object=column_object,
                        class_name=COLUMN_CLASS_NAME,
                        uuid=column_uuid
                    )
                    # Add reference to parent table
                    batch.add_reference(
                        from_object_uuid=column_uuid,
                        from_object_class_name=COLUMN_CLASS_NAME,
                        from_property_name="fromTable",
                        to_object_uuid=table_uuid
                    )
                    print(f"  🔸 Queued Column: {column_name} -> {table_name}")
                else:
                    print(f"  🔸 Column '{column_name}' in table '{table_name}' already exists, skipping.")

    print("✅ Data import (append) completed.")

def print_tables_and_columns(client):
    print("\nTables and their columns stored in Weaviate:")
    # Get all tables
    tables = client.query.get("PwcTable", ["tableName"]).with_limit(100).do()
    table_names = [t["tableName"] for t in tables["data"]["Get"]["PwcTable"]]
    for table in table_names:
        print(f"Table: {table}")
        # Get columns for this table
        columns = client.query.get("PwcColumn", ["columnName"]).with_where({
            "path": ["tableName"],
            "operator": "Equal",
            "valueText": table
        }).with_limit(100).do()
        col_names = [c["columnName"] for c in columns["data"]["Get"]["PwcColumn"]]
        print("  Columns:", ", ".join(col_names))

if __name__ == "__main__":
    main()
    # After main ingestion, print all tables and columns
    # try:
    #     client = weaviate.Client(
    #         url=WEAVIATE_URL,
    #         auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    #         additional_headers={"X-HuggingFace-Api-Key": HF_API_KEY}
    #     )
    #     print_tables_and_columns(client)
    # except Exception as e:
    #     print(f"Could not print tables and columns: {e}")