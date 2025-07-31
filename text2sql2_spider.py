import weaviate
import json
import os
import re
from dotenv import load_dotenv

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    raise ImportError("langchain is not installed. Please install it with 'pip install langchain'.")

# --- Load environment variables and Constants ---
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TABLE_CLASS_NAME = "PwcTable"
COLUMN_CLASS_NAME = "PwcColumn"
DB_FILE = "/Users/mdnazisharman/ngst/gst_data.db"
GEMINI_MODEL = "gemini-2.5-pro"

def init_clients():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
    openai_api_key = OPENAI_API_KEY
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is None. Please set it in your .env file.")

    from weaviate.auth import AuthApiKey
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={"X-HuggingFace-Api-Key": HF_API_KEY}
    )
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        api_version="2024-02-15-preview",
        temperature=0.0,
        max_tokens=1000,
        openai_api_key=openai_api_key, # type: ignore
        azure_endpoint="https://nextiva.openai.azure.com/",
        streaming=True,
    )
    return client, llm

def clean_sql_query(raw_sql):
    sql = re.sub(r"```sql\n|```", "", raw_sql).strip()
    return sql

def retrieve_tables(client, question, column_limit=30):
    try:
        column_response = (
            client.query
            .get(COLUMN_CLASS_NAME, ["tableName", "columnName"])
            .with_hybrid(query=question, alpha=0.75)
            .with_limit(column_limit)
            .do()
        )
        candidate_columns = column_response['data']['Get'][COLUMN_CLASS_NAME]
    except Exception as e:
        print("--- DEBUG: EXITING retrieve_tables (DUE TO ERROR) ---\n")
        return []

    if not candidate_columns:
        print("No candidate columns found by Weaviate search.")
        print("--- DEBUG: EXITING retrieve_tables (NO COLUMNS FOUND) ---\n")
        return []

    columns_by_table = {}
    for col in candidate_columns:
        columns_by_table.setdefault(col['tableName'], set()).add(col['columnName'])

    table_names = list(columns_by_table.keys())
    where_filter = {"path": ["tableName"], "operator": "ContainsAny", "valueText": table_names}
    
    try:
        table_response = (
            client.query
            .get(TABLE_CLASS_NAME, ["tableSummary", "fullMetadata"])
            .with_where(where_filter)
            .do()
        )
        retrieved_tables_full_meta = table_response['data']['Get'][TABLE_CLASS_NAME]
    except Exception as e:
        print("--- DEBUG: EXITING retrieve_tables (DUE TO ERROR) ---\n")
        return []

    final_results = []
    for table_data in retrieved_tables_full_meta:
        full_meta_obj = json.loads(table_data['fullMetadata'])
        table_name = full_meta_obj['table_name']
        relevant_column_names = columns_by_table.get(table_name, set())
        
        filtered_columns = [
            col_meta for col_meta in full_meta_obj.get('columns', [])
            if col_meta.get('name') in relevant_column_names
        ]

        if not filtered_columns: continue

        filtered_metadata_obj = {
            "table_name": table_name,
            "description": full_meta_obj.get('description', ''),
            "columns": filtered_columns
        }
        final_results.append({
            "summary": table_data['tableSummary'],
            "metadata_information": json.dumps(filtered_metadata_obj)
        })

    return final_results

def create_ddl_from_metadata(meta):
    table_name = meta["table_name"]
    columns = meta["columns"]
    col_defs = [f"  `{col['name']}` {col['type']} -- {col.get('description', '')}" for col in columns]
    return f"CREATE TABLE `{table_name}` (\n  " + ",\n  ".join(col_defs) + "\n);"

def extract_used_schema(model, question, tables, return_prompt=False):
    metadata_map = {json.loads(t['metadata_information'])['table_name']: json.loads(t['metadata_information']) for t in tables}
    if not metadata_map: 
        if return_prompt:
            return [], ""
        return []
    schema = "\n\n".join([create_ddl_from_metadata(table) for table in metadata_map.values()])
    prompt = f"""
        You are an expert SQL assistant. Based on the user's question, the provided evidence, and the full database schema, identify the specific tables and columns required.
        === Question ===
        {question}
        === Full Schema ===
        {schema}
        === Response Format ===
        Return a JSON object with a single key \"tables\", which is a list of objects, each with \"table_name\" and a \"columns\" list.
        Example: {{"tables": [{{"table_name": "employees", "columns": ["employee_id", "status"]}}]}}
    """
    response = model.invoke(prompt)
    match = re.search(r'{.*}', response.content, re.DOTALL)
    if match:
        try: 
            tables = json.loads(match.group())['tables']
            if return_prompt:
                return tables, prompt
            return tables
        except (json.JSONDecodeError, KeyError): 
            if return_prompt:
                return [], prompt
            return []
    if return_prompt:
        return [], prompt
    return []

def generate_sql(model, question, filtered_schema, all_tables_metadata, return_prompt=False):
    selected_table_names = [t['table_name'] for t in filtered_schema]
    ddl_statements = [create_ddl_from_metadata(json.loads(meta_str)) for meta_str in all_tables_metadata if json.loads(meta_str)['table_name'] in selected_table_names]
    schema_ddl = "\n\n".join(ddl_statements)
    
    prompt = f"""
        You are an expert data analyst who writes flawless, human-readable SQL for a SQLite database.
        Your task is to answer the user's question by generating a single, valid SQL query.
        Adhere to these strict rules:
        1.  The output must be a SINGLE, executable SQL query.
        2.  Use ONLY the tables and columns listed in the "Relevant Database Schema". Restriction: You cannot use any other tables or columns that are not listed in the "Relevant Database Schema".
        3.  If using Common Table Expressions (CTEs), they must all be part of a single query initiated with the `WITH` keyword. The final `SELECT` statement should then use these CTEs to produce the result.
        4.  **Be conservative with `WHERE` clauses.** Do not add filters for columns like `documentType` or `status` unless the user's question explicitly mentions them (e.g., "show me only invoices", "where status is validated").
        5.  Use user-friendly aliases for column names (e.g., `productServiceDescription AS "Product Description"`).
        6.  Return ONLY the SQL query/script. Do not include any explanations or markdown formatting.
        7.  Always return a valid query.
        === User Question ===
        {question}
        === Relevant Database Schema (DDL) ===
        {schema_ddl}
        SQL Query:
    """
    response = model.invoke(prompt)
    if return_prompt:
        return clean_sql_query(response.content), prompt
    return clean_sql_query(response.content)

if __name__ == "__main__":
    # Read the spider_natural_queries.jsonl file
    input_jsonl = "spider_natural_queries.jsonl"
    output_dir = "spider_sqls"
    os.makedirs(output_dir, exist_ok=True)

    # 0. Initialize clients
    client, model = init_clients()
    print("✅ Clients initialized.")

    # Read each line (instance) from the .jsonl file
    with open(input_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                print(f"Skipping line due to JSON error: {e}")
                continue

            # Extract question and instance_id
            question = item.get("question", "")
            instance_id = item.get("instance_id", "")
            # evidence = item.get("evidence", "")

            if not question or not instance_id:
                print("Skipping item due to missing question or instance_id.")
                continue

            print(f"\nProcessing instance_id: {instance_id}")
            try:
                # 1. Retrieve relevant table metadata from Weaviate
                tables_metadata = retrieve_tables(client, question)
                if not tables_metadata:
                    print(f"❌ ERROR: No relevant tables found in Weaviate for instance_id {instance_id}. Skipping.")
                    continue

                # 2. Use LLM to extract the precise schema needed
                filtered_schema, _ = extract_used_schema(model, question, tables_metadata, return_prompt=True)
                if not filtered_schema:
                    print(f"❌ ERROR: LLM could not determine a schema for instance_id {instance_id}. Skipping.")
                    continue

                # 3. Generate SQL based on the filtered schema
                all_meta_strings = [t['metadata_information'] for t in tables_metadata]
                sql_script, _ = generate_sql(model, question, filtered_schema, all_meta_strings, return_prompt=True)
                sql_script = sql_script.strip()

                # 4. Write the SQL to the output file
                output_path = os.path.join(output_dir, f"{instance_id}.sql")
                with open(output_path, "w") as out_f:
                    out_f.write(sql_script + "\n")
                print(f"✅ SQL written to {output_path}")

            except Exception as e:
                print(f"FATAL ERROR for instance_id {instance_id}: {e}")

    print("All SQL files written to spider_sqls/")
