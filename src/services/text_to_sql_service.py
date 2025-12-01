"""
Text-to-SQL Service for AI Customer Agent.

This module provides the Text-to-SQL service that converts natural language queries
about Excel data into SQL queries using DeepSeek API, and executes them on the
SQLite database containing Excel data as required by US-010.
"""

import re
import logging
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from ..services.deepseek_service import DeepSeekService
from ..services.sqlite_database_service import SQLiteDatabaseService
from ..models.excel_models import ExcelDocument, ExcelSheetData


class TextToSQLError(Exception):
    """Custom exception for Text-to-SQL service related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None, original_query: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        self.original_query = original_query
        super().__init__(self.message)


class TextToSQLService:
    """
    Service for converting natural language queries to SQL and executing them.
    
    This service uses DeepSeek API to generate SQL queries from natural language
    and executes them on the SQLite database containing Excel data.
    """
    
    def __init__(self, deepseek_service: DeepSeekService, sqlite_service: SQLiteDatabaseService):
        """
        Initialize the Text-to-SQL service.
        
        Args:
            deepseek_service: DeepSeekService instance for SQL generation
            sqlite_service: SQLiteDatabaseService instance for query execution
        """
        self.deepseek_service = deepseek_service
        self.sqlite_service = sqlite_service
        self.logger = logging.getLogger(__name__)
        
    async def convert_to_sql(self, natural_language_query: str, file_id: str, 
                           sheet_name: Optional[str] = None, 
                           conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Convert natural language query to SQL and execute it.
        
        Args:
            natural_language_query: Natural language query about Excel data
            file_id: ID of the Excel file to query
            sheet_name: Optional specific sheet name to query
            conversation_context: Optional conversation history for context
            
        Returns:
            Dictionary containing SQL query, results, and metadata
            
        Raises:
            TextToSQLError: If conversion or execution fails
        """
        try:
            self.logger.info(f"Converting natural language query to SQL: {natural_language_query}")
            
            # Get schema information for the Excel file
            schema_info = await self._get_schema_info(file_id, sheet_name)
            
            # Generate SQL query using DeepSeek API
            sql_query = await self._generate_sql_query(
                natural_language_query, 
                schema_info, 
                conversation_context
            )
            
            # Validate and clean SQL query
            cleaned_sql = self._clean_sql_query(sql_query)
            
            # Execute SQL query
            results = self._execute_sql_query(cleaned_sql, file_id, sheet_name)
            
            # Format results for better readability
            formatted_results = self._format_sql_results(results)
            
            return {
                "original_query": natural_language_query,
                "sql_query": cleaned_sql,
                "results": formatted_results,
                "result_count": len(results),
                "file_id": file_id,
                "sheet_name": sheet_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Text-to-SQL conversion failed: {str(e)}")
            raise TextToSQLError(
                f"Failed to convert query to SQL: {str(e)}",
                error_type="conversion_error",
                original_query=natural_language_query
            )
            
    async def _get_schema_info(self, file_id: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get schema information for Excel file including actual table names from dynamic tables.
        
        Args:
            file_id: ID of the Excel file
            sheet_name: Optional specific sheet name
            
        Returns:
            Dictionary containing schema information with actual table names
            
        Raises:
            TextToSQLError: If schema discovery fails
        """
        try:
            # Get Excel file metadata
            excel_file = self.sqlite_service.get_excel_file(file_id)
            if not excel_file:
                raise TextToSQLError(f"Excel file not found: {file_id}")
                
            # Get dynamic table schemas for this file
            table_schemas = self.sqlite_service.get_dynamic_table_schemas(file_id)
            if not table_schemas:
                raise TextToSQLError(f"No dynamic tables found for file: {file_id}")
                
            # Get sheet data for additional context
            sheet_data_list = self.sqlite_service.get_sheet_data(file_id, sheet_name)
                
            schema_info = {
                "file_name": excel_file.file_name,
                "file_id": file_id,
                "actual_tables": {},
                "sheets": []
            }
            
            # Add actual table information from dynamic tables
            for table_name, table_info in table_schemas.items():
                schema_info["actual_tables"][table_name] = {
                    "columns": table_info["columns"],
                    "row_count": table_info["row_count"],
                    "file_id": table_info["file_id"]
                }
            
            # Add sheet information for context (backward compatibility)
            for sheet_data in sheet_data_list:
                sheet_info = {
                    "sheet_name": sheet_data.sheet_name,
                    "headers": sheet_data.headers,
                    "row_count": sheet_data.row_count,
                    "column_count": sheet_data.column_count,
                    "data_types": sheet_data.data_types,
                    "sample_data": sheet_data.sample_data[:3],  # First 3 rows for context
                    # Map sheet name to actual table name
                    "actual_table_name": f"excel_{file_id.replace('excel_', '')}_{sheet_data.sheet_name.lower()}"
                }
                schema_info["sheets"].append(sheet_info)
                
            self.logger.info(f"Retrieved schema info for file {file_id} with {len(table_schemas)} dynamic tables")
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Failed to get schema info: {str(e)}")
            raise TextToSQLError(f"Schema discovery failed: {str(e)}")
            
    async def _generate_sql_query(self, natural_language_query: str, 
                                schema_info: Dict[str, Any],
                                conversation_context: Optional[List[Dict]] = None) -> str:
        """
        Generate SQL query from natural language using DeepSeek API.
        
        Args:
            natural_language_query: Natural language query
            schema_info: Schema information about the Excel data
            conversation_context: Optional conversation history
            
        Returns:
            Generated SQL query string
        """
        # Build system prompt for SQL generation
        system_prompt = self._build_sql_generation_prompt(schema_info)
        
        # Build messages for API call
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation context if available
        if conversation_context:
            messages.extend(conversation_context)
            
        # Add the current query
        messages.append({
            "role": "user", 
            "content": f"Query: {natural_language_query}\n\nPlease generate a SQL query for this question."
        })
        
        try:
            # Call DeepSeek API to generate SQL
            response = await self.deepseek_service.chat_completion(messages)
            
            # Extract SQL query from response
            sql_query = self._extract_sql_from_response(response)
            
            self.logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            raise TextToSQLError(f"SQL generation failed: {str(e)}")
            
    def _build_sql_generation_prompt(self, schema_info: Dict[str, Any]) -> str:
        """
        Build system prompt for SQL generation with actual dynamic table names.
        
        Args:
            schema_info: Schema information about the Excel data including actual table names
            
        Returns:
            System prompt string with actual table information
        """
        prompt = f"""
You are an expert SQL query generator for Excel data. Your task is to convert natural language questions into SQL queries that can be executed on the actual SQLite database tables containing Excel data.

DATABASE SCHEMA INFORMATION:
File: {schema_info['file_name']}
File ID: {schema_info['file_id']}

AVAILABLE TABLES AND THEIR SCHEMAS:
"""
        
        # Add actual table information from dynamic tables
        for table_name, table_info in schema_info["actual_tables"].items():
            prompt += f"""
Table: {table_name}
- Row Count: {table_info['row_count']}
- Columns:
"""
            for column in table_info["columns"]:
                prompt += f"  - {column['name']} ({column['type']})"
                if column.get('primary_key'):
                    prompt += " [PRIMARY KEY]"
                prompt += "\n"
        
        # Also include sheet information for context
        prompt += "\nSHEET INFORMATION (for reference):\n"
        for sheet in schema_info["sheets"]:
            prompt += f"""
Sheet: {sheet['sheet_name']}
- Actual Table: {sheet.get('actual_table_name', 'N/A')}
- Headers: {', '.join(sheet['headers'])}
- Data Types: {json.dumps(sheet['data_types'], indent=2)}
- Sample Data (first 3 rows):
{json.dumps(sheet['sample_data'], indent=2)}
"""
        
        prompt += """
IMPORTANT RULES:
1. The data is stored in ACTUAL SQLite tables with the following structure:
   - Each Excel sheet is stored as a separate relational table in SQLite
   - Table names follow the pattern: "excel_{file_id}_{sheet_name}" (use exact names from above)
   - Columns use sanitized header names (spaces and special characters removed)
   - Data types are properly mapped to SQLite types (TEXT, INTEGER, REAL, etc.)

2. SQL QUERY GUIDELINES:
   - Use standard SQL syntax compatible with SQLite
   - Always use proper quoting for table and column names with double quotes
   - Use appropriate data type handling (strings, numbers, dates)
   - Include WHERE clauses for filtering when needed
   - Use aggregate functions (COUNT, SUM, AVG, MAX, MIN) for statistical questions
   - Use ORDER BY for sorting results
   - Use LIMIT to restrict results when appropriate
   - For column names with spaces or special characters, use double quotes: "column name"

3. RESPONSE FORMAT:
   - Return ONLY the SQL query without any explanation
   - Do not include markdown formatting or code blocks
   - Ensure the query is syntactically correct and executable
   - Use the exact table names provided above

4. COMMON PATTERNS:
   - For counting records: SELECT COUNT(*) FROM "table_name"
   - For finding specific values: SELECT * FROM "table_name" WHERE "column_name" = 'value'
   - For numerical analysis: SELECT AVG("column_name"), MAX("column_name") FROM "table_name"
   - For date filtering: SELECT * FROM "table_name" WHERE "date_column" >= '2024-01-01'
   - For grouping: SELECT "category", COUNT(*) FROM "table_name" GROUP BY "category"

Example queries:
- "How many rows are in the sales sheet?" → SELECT COUNT(*) FROM "excel_file123_sales"
- "Show me all products with price greater than 100" → SELECT * FROM "excel_file123_products" WHERE "Price" > 100
- "What is the average revenue by month?" → SELECT "Month", AVG("Revenue") FROM "excel_file123_sales" GROUP BY "Month"

Now generate the SQL query for the user's question.
"""
        return prompt
        
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL query from DeepSeek API response.
        
        Args:
            response: API response string
            
        Returns:
            Extracted SQL query
        """
        # Remove markdown code blocks if present
        response = re.sub(r'```sql\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Remove any explanatory text and get just the SQL
        lines = response.strip().split('\n')
        sql_lines = []
        
        # More robust SQL extraction that preserves multi-line structure
        sql_started = False
        for line in lines:
            line = line.strip()
            if not line:
                # If we've started SQL and hit a blank line, stop collecting
                if sql_started:
                    break
                continue
                
            # Check if this line contains SQL keywords
            has_sql_keyword = (
                line.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')) or
                any(keyword in line.upper() for keyword in ['FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT'])
            )
            
            # If we haven't started SQL yet but found a SQL keyword, start collecting
            if not sql_started and has_sql_keyword:
                sql_started = True
                
            # If we're in SQL mode, collect the line
            if sql_started:
                sql_lines.append(line)
        
        sql_query = ' '.join(sql_lines)
        
        # If no SQL found, use the entire response
        if not sql_query:
            sql_query = response.strip()
            
        return sql_query
        
    def _clean_sql_query(self, sql_query: str) -> str:
        """
        Clean and validate SQL query.
        
        Args:
            sql_query: Raw SQL query string
            
        Returns:
            Cleaned SQL query
        """
        # Remove trailing semicolons and whitespace
        sql_query = sql_query.rstrip(';').strip()
        
        # Basic SQL injection prevention
        dangerous_patterns = [
            r'\bDROP\b',
            r'\bDELETE\b.*\bFROM\b',
            r'\bUPDATE\b.*\bSET\b',
            r'\bINSERT\b.*\bINTO\b',
            r'\bALTER\b',
            r'\bCREATE\b',
            r'\bTRUNCATE\b',
            r';.*;',  # Multiple statements
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                raise TextToSQLError(
                    "Query contains potentially dangerous operations",
                    error_type="security_error"
                )
                
        return sql_query
        
    def _execute_sql_query(self, sql_query: str, file_id: str, 
                          sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query on actual relational tables in SQLite database.
        
        This method uses the SQLiteDatabaseService to execute the generated SQL query
        directly on the dynamic tables created from Excel sheets, providing actual
        SQL execution on relational data as required by US-010.
        
        Args:
            sql_query: SQL query to execute on relational tables
            file_id: ID of the Excel file for context and validation
            sheet_name: Optional specific sheet name for validation
            
        Returns:
            List of query results from actual SQL execution
            
        Raises:
            TextToSQLError: If SQL execution fails or returns no results
        """
        try:
            # Validate that the file exists and has dynamic tables
            excel_file = self.sqlite_service.get_excel_file(file_id)
            if not excel_file:
                raise TextToSQLError(f"Excel file not found: {file_id}")
                
            # Get dynamic table schemas to validate the file has tables
            table_schemas = self.sqlite_service.get_dynamic_table_schemas(file_id)
            if not table_schemas:
                raise TextToSQLError(f"No dynamic tables found for file: {file_id}")
            
            # Execute the SQL query using the SQLiteDatabaseService
            # This performs actual SQL execution on relational tables
            results = self.sqlite_service.execute_sql_query(sql_query)
            
            # Validate that we got results (empty results are valid for some queries)
            if results is None:
                raise TextToSQLError("SQL query execution returned no results")
                
            self.logger.info(f"Successfully executed SQL query on relational tables, returned {len(results)} results")
            return results
                
        except Exception as e:
            self.logger.error(f"SQL execution on relational tables failed: {str(e)}")
            raise TextToSQLError(
                f"SQL execution failed: {str(e)}",
                error_type="execution_error"
            )
            
    def _format_sql_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format SQL results for better readability.
        
        Args:
            results: Raw SQL results
            
        Returns:
            Formatted results
        """
        formatted_results = []
        
        for result in results:
            formatted_result = {}
            for key, value in result.items():
                # Handle different data types for display
                if value is None:
                    formatted_result[key] = None
                elif isinstance(value, (int, float)):
                    formatted_result[key] = value
                elif isinstance(value, str):
                    # Try to parse as JSON if it looks like JSON
                    if value.startswith('{') and value.endswith('}'):
                        try:
                            formatted_result[key] = json.loads(value)
                        except json.JSONDecodeError:
                            formatted_result[key] = value
                    else:
                        formatted_result[key] = value
                else:
                    formatted_result[key] = str(value)
                    
            formatted_results.append(formatted_result)
            
        return formatted_results
        
    async def explain_sql_query(self, sql_query: str, natural_language_query: str) -> Dict[str, Any]:
        """
        Explain what a SQL query does in natural language.
        
        Args:
            sql_query: SQL query to explain
            natural_language_query: Original natural language query
            
        Returns:
            Dictionary containing explanation
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert SQL explainer. Your task is to explain what a SQL query does in simple, natural language.

Please provide:
1. A brief summary of what the query does
2. Key operations being performed (SELECT, WHERE, GROUP BY, etc.)
3. What the results will show
4. Any important limitations or assumptions

Keep the explanation clear and concise, suitable for non-technical users."""
                },
                {
                    "role": "user",
                    "content": f"Original question: {natural_language_query}\n\nSQL Query: {sql_query}\n\nPlease explain what this SQL query does."
                }
            ]
            
            explanation = await self.deepseek_service.chat_completion(messages)
            
            return {
                "sql_query": sql_query,
                "original_query": natural_language_query,
                "explanation": explanation.strip(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"SQL explanation failed: {str(e)}")
            raise TextToSQLError(f"SQL explanation failed: {str(e)}")
            
    def get_available_tables(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Get available tables (sheets) for an Excel file.
        
        Args:
            file_id: ID of the Excel file
            
        Returns:
            List of table information
        """
        try:
            sheet_data_list = self.sqlite_service.get_sheet_data(file_id)
            
            tables = []
            for sheet_data in sheet_data_list:
                table_info = {
                    "sheet_name": sheet_data.sheet_name,
                    "columns": [],
                    "row_count": sheet_data.row_count,
                    "sample_data": sheet_data.sample_data[:5]  # First 5 rows
                }
                
                for header in sheet_data.headers:
                    column_info = {
                        "name": header,
                        "data_type": sheet_data.data_types.get(header, "unknown"),
                        "description": f"Column from {sheet_data.sheet_name} sheet"
                    }
                    table_info["columns"].append(column_info)
                    
                tables.append(table_info)
                
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to get available tables: {str(e)}")
            raise TextToSQLError(f"Failed to get available tables: {str(e)}")
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Text-to-SQL service.
        
        Returns:
            Dictionary containing health status
        """
        try:
            # Check DeepSeek service
            deepseek_healthy = await self.deepseek_service.health_check()
            
            # Check SQLite service
            sqlite_healthy = self.sqlite_service.health_check()
            
            return {
                "service": "text_to_sql",
                "status": "healthy" if (deepseek_healthy and sqlite_healthy) else "unhealthy",
                "deepseek_service": "healthy" if deepseek_healthy else "unhealthy",
                "sqlite_service": "healthy" if sqlite_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "service": "text_to_sql",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
