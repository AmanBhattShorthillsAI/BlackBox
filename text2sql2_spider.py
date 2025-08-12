import weaviate
import json
import os
import re
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    raise ImportError("langchain is not installed. Please install it with 'pip install langchain langchain-community'.")

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
            return [], "", {}
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
    
    # Get response and track token usage
    response = model.invoke(prompt)
    
    # Extract token info from response metadata
    token_info = {}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        token_info = {
            "prompt_tokens": response.usage_metadata.get('input_tokens', 0),
            "completion_tokens": response.usage_metadata.get('output_tokens', 0),
            "total_tokens": response.usage_metadata.get('input_tokens', 0) + response.usage_metadata.get('output_tokens', 0)
        }
    
    match = re.search(r'{.*}', response.content, re.DOTALL)
    if match:
        try: 
            tables = json.loads(match.group())['tables']
            if return_prompt:
                return tables, prompt, token_info
            return tables
        except (json.JSONDecodeError, KeyError): 
            if return_prompt:
                return [], prompt, token_info
            return []
    if return_prompt:
        return [], prompt, token_info
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
    
    # Get response and track token usage
    response = model.invoke(prompt)
    
    # Extract token info from response metadata
    token_info = {}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        token_info = {
            "prompt_tokens": response.usage_metadata.get('input_tokens', 0),
            "completion_tokens": response.usage_metadata.get('output_tokens', 0),
            "total_tokens": response.usage_metadata.get('input_tokens', 0) + response.usage_metadata.get('output_tokens', 0)
        }
    
    if return_prompt:
        return clean_sql_query(response.content), prompt, token_info
    return clean_sql_query(response.content)

def extract_database_names(tables_metadata):
    """Extract unique database names from the tables metadata"""
    db_names = set()
    for table in tables_metadata:
        meta = json.loads(table['metadata_information'])
        # Assuming database name is part of table name or we use a default
        # You might need to adjust this based on your actual schema
        table_name = meta.get('table_name', '')
        # If table names have database prefix like 'db_name.table_name'
        if '.' in table_name:
            db_name = table_name.split('.')[0]
            db_names.add(db_name)
        else:
            # Otherwise, we might infer from the metadata or use a default
            db_names.add('spider_db')  # Default database name
    
    return ', '.join(sorted(db_names)) if db_names else 'unknown'

def append_to_excel(excel_filename, instance_metrics):
    """Append a single row to the Excel file efficiently"""
    # Read existing data
    df = pd.read_excel(excel_filename)
    
    # Append new row
    new_row = pd.DataFrame([instance_metrics])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Write back with formatting
    with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Auto-adjust column widths for better readability
        worksheet = writer.sheets['Metrics']
        for column in worksheet.columns:
            max_length = 0
            column_cells = [cell for cell in column]
            for cell in column_cells:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_cells[0].column_letter].width = adjusted_width

if __name__ == "__main__":
    # Read the spider_natural_queries.jsonl file
    input_jsonl = "spider_natural_queries.jsonl"
    output_dir = "spider_sqls"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"spider_sql_generation_metrics_{timestamp}.xlsx"
    
    # Initialize Excel file with headers
    initial_data = pd.DataFrame(columns=[
        "instance_id",
        "question",
        "database_used",
        "input_tokens",
        "output_tokens",
        "sql_generation_time",
        "status",
        "error_message",
        "generated_sql",
        "schema_extraction_prompt",
        "sql_generation_prompt"
    ])
    
    # Save initial Excel file
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        initial_data.to_excel(writer, sheet_name='Metrics', index=False)
    
    print(f"📊 Created metrics file: {excel_filename}")

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

            # Extract question, instance_id, and database
            question = item.get("question", "")
            instance_id = item.get("instance_id", "")
            database = item.get("db", "unknown")
            # evidence = item.get("evidence", "")

            if not question or not instance_id:
                print("Skipping item due to missing question or instance_id.")
                continue

            print(f"\nProcessing instance_id: {instance_id}")
            
            # Initialize metrics for this instance
            instance_metrics = {
                "instance_id": instance_id,
                "question": question,
                "schema_extraction_prompt": "",
                "sql_generation_prompt": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "sql_generation_time": 0,
                "database_used": database,
                "status": "success",
                "error_message": "",
                "generated_sql": ""
            }
            
            try:
                # Start timing
                start_time = time.time()
                
                # 1. Retrieve relevant table metadata from Weaviate
                tables_metadata = retrieve_tables(client, question)
                if not tables_metadata:
                    instance_metrics["status"] = "error"
                    instance_metrics["error_message"] = "No relevant tables found in Weaviate"
                    # Update Excel file incrementally
                    append_to_excel(excel_filename, instance_metrics)
                    print(f"❌ ERROR: No relevant tables found in Weaviate for instance_id {instance_id}. Skipping.")
                    continue

                # 2. Use LLM to extract the precise schema needed
                filtered_schema, schema_prompt, schema_tokens = extract_used_schema(model, question, tables_metadata, return_prompt=True)
                instance_metrics["schema_extraction_prompt"] = schema_prompt
                
                # Accumulate tokens
                instance_metrics["input_tokens"] += schema_tokens.get("prompt_tokens", 0)
                instance_metrics["output_tokens"] += schema_tokens.get("completion_tokens", 0)
                
                if not filtered_schema:
                    instance_metrics["status"] = "error"
                    instance_metrics["error_message"] = "LLM could not determine a schema"
                    # Update Excel file incrementally
                    append_to_excel(excel_filename, instance_metrics)
                    print(f"❌ ERROR: LLM could not determine a schema for instance_id {instance_id}. Skipping.")
                    continue

                # 3. Generate SQL based on the filtered schema
                all_meta_strings = [t['metadata_information'] for t in tables_metadata]
                sql_script, sql_prompt, sql_tokens = generate_sql(model, question, filtered_schema, all_meta_strings, return_prompt=True)
                sql_script = sql_script.strip()
                
                # End timing
                end_time = time.time()
                
                instance_metrics["sql_generation_prompt"] = sql_prompt
                
                # Accumulate tokens for final total
                instance_metrics["input_tokens"] += sql_tokens.get("prompt_tokens", 0)
                instance_metrics["output_tokens"] += sql_tokens.get("completion_tokens", 0)
                instance_metrics["sql_generation_time"] = round(end_time - start_time, 2)
                instance_metrics["generated_sql"] = sql_script

                # 4. Write the SQL to the output file
                output_path = os.path.join(output_dir, f"{instance_id}.sql")
                with open(output_path, "w") as out_f:
                    out_f.write(sql_script + "\n")
                print(f"✅ SQL written to {output_path}")
                
                # Update Excel file incrementally
                append_to_excel(excel_filename, instance_metrics)

            except Exception as e:
                instance_metrics["status"] = "error"
                instance_metrics["error_message"] = str(e)
                # Update Excel file incrementally
                append_to_excel(excel_filename, instance_metrics)
                print(f"FATAL ERROR for instance_id {instance_id}: {e}")

    # Final summary
    print(f"\n✅ All SQL files written to {output_dir}/")
    print(f"✅ Metrics continuously saved to {excel_filename}")
    
    # Read final metrics for summary
    final_df = pd.read_excel(excel_filename)
    print(f"\nSummary:")
    print(f"- Total instances processed: {len(final_df)}")
    print(f"- Successful: {len(final_df[final_df['status'] == 'success'])}")
    print(f"- Errors: {len(final_df[final_df['status'] == 'error'])}")
    if len(final_df) > 0:
        avg_time = final_df['sql_generation_time'].mean()
        print(f"- Average SQL generation time: {avg_time:.2f} seconds")
        print(f"- Total input tokens: {final_df['input_tokens'].sum():,}")
        print(f"- Total output tokens: {final_df['output_tokens'].sum():,}")
