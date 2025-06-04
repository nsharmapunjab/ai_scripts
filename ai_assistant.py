#!/usr/bin/env python3
"""
AI Assistant with Model Context Protocol (MCP) Implementation
A beginner-friendly tutorial for building an AI-powered backend

This script demonstrates:
1. How to create MCP-style tool integration
2. How to connect AI with external data sources
3. How AI can intelligently choose which tools to use
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DataAccessTool:
    """
    This tool handles reading and querying our customer data.
    Think of it as a smart filing cabinet that can answer questions about customers.
    """
    
    def __init__(self, csv_file_path: str):
        print("📁 Loading customer data...")
        self.data = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(self.data)} customer records")
        
    def get_tool_description(self) -> Dict:
        """
        This tells the AI what this tool can do - like a job description for the tool
        """
        return {
            "name": "query_customer_data",
            "description": "Search and retrieve customer information from the database. Can filter by customer name, category, or get statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["find_customer", "category_stats", "top_customers", "all_data"],
                        "description": "Type of query to perform"
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Value to filter by (customer name, category, etc.)"
                    }
                },
                "required": ["query_type"]
            }
        }
    
    def execute(self, query_type: str, filter_value: str = None) -> str:
        """
        This is where the actual work happens - like an employee doing their job
        """
        print(f"🔍 Executing data query: {query_type}")
        
        try:
            if query_type == "find_customer":
                if not filter_value:
                    return "Please provide a customer name to search for."
                
                # Search for customer by name (case-insensitive)
                customer = self.data[self.data['name'].str.contains(filter_value, case=False, na=False)]
                if customer.empty:
                    return f"No customer found with name containing '{filter_value}'"
                
                result = customer.iloc[0]  # Get first match
                return f"Customer Found:\nName: {result['name']}\nEmail: {result['email']}\nTotal Purchases: ${result['purchase_amount']}\nFavorite Category: {result['product_category']}\nJoined: {result['join_date']}"
            
            elif query_type == "category_stats":
                if filter_value:
                    # Stats for specific category
                    category_data = self.data[self.data['product_category'].str.contains(filter_value, case=False, na=False)]
                    if category_data.empty:
                        return f"No data found for category '{filter_value}'"
                    
                    total_sales = category_data['purchase_amount'].sum()
                    avg_purchase = category_data['purchase_amount'].mean()
                    customer_count = len(category_data)
                    
                    return f"Category: {filter_value}\nTotal Sales: ${total_sales:.2f}\nAverage Purchase: ${avg_purchase:.2f}\nNumber of Customers: {customer_count}"
                else:
                    # Overall category breakdown
                    category_stats = self.data.groupby('product_category')['purchase_amount'].agg(['sum', 'mean', 'count'])
                    result = "Category Statistics:\n"
                    for category, stats in category_stats.iterrows():
                        result += f"{category}: ${stats['sum']:.2f} total, ${stats['mean']:.2f} avg, {stats['count']} customers\n"
                    return result
            
            elif query_type == "top_customers":
                top_customers = self.data.nlargest(3, 'purchase_amount')
                result = "Top 3 Customers by Purchase Amount:\n"
                for idx, customer in top_customers.iterrows():
                    result += f"{customer['name']}: ${customer['purchase_amount']:.2f}\n"
                return result
            
            elif query_type == "all_data":
                total_customers = len(self.data)
                total_revenue = self.data['purchase_amount'].sum()
                avg_purchase = self.data['purchase_amount'].mean()
                
                return f"Database Summary:\nTotal Customers: {total_customers}\nTotal Revenue: ${total_revenue:.2f}\nAverage Purchase: ${avg_purchase:.2f}"
            
            else:
                return f"Unknown query type: {query_type}"
                
        except Exception as e:
            return f"Error executing query: {str(e)}"


class CalculatorTool:
    """
    This tool handles mathematical calculations.
    Think of it as a smart calculator that can do various math operations.
    """
    
    def get_tool_description(self) -> Dict:
        """
        Describes what this calculator tool can do
        """
        return {
            "name": "calculate",
            "description": "Perform mathematical calculations including basic arithmetic, percentages, and business calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "percentage", "discount"],
                        "description": "Type of calculation to perform"
                    },
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numbers to calculate with"
                    },
                    "percentage": {
                        "type": "number",
                        "description": "Percentage value for percentage or discount calculations"
                    }
                },
                "required": ["operation", "numbers"]
            }
        }
    
    def execute(self, operation: str, numbers: List[float], percentage: float = None) -> str:
        """
        Performs the actual calculations
        """
        print(f"🧮 Performing calculation: {operation} with numbers {numbers}")
        
        try:
            if operation == "add":
                result = sum(numbers)
                return f"Addition result: {' + '.join(map(str, numbers))} = {result}"
            
            elif operation == "subtract":
                result = numbers[0]
                for num in numbers[1:]:
                    result -= num
                return f"Subtraction result: {' - '.join(map(str, numbers))} = {result}"
            
            elif operation == "multiply":
                result = 1
                for num in numbers:
                    result *= num
                return f"Multiplication result: {' × '.join(map(str, numbers))} = {result}"
            
            elif operation == "divide":
                if len(numbers) != 2:
                    return "Division requires exactly 2 numbers"
                if numbers[1] == 0:
                    return "Cannot divide by zero"
                result = numbers[0] / numbers[1]
                return f"Division result: {numbers[0]} ÷ {numbers[1]} = {result}"
            
            elif operation == "percentage":
                if len(numbers) != 1 or percentage is None:
                    return "Percentage calculation requires 1 number and a percentage value"
                result = (numbers[0] * percentage) / 100
                return f"Percentage calculation: {percentage}% of {numbers[0]} = {result}"
            
            elif operation == "discount":
                if len(numbers) != 1 or percentage is None:
                    return "Discount calculation requires 1 number (original price) and discount percentage"
                discount_amount = (numbers[0] * percentage) / 100
                final_price = numbers[0] - discount_amount
                return f"Discount calculation: ${numbers[0]} with {percentage}% discount = ${final_price:.2f} (saved ${discount_amount:.2f})"
            
            else:
                return f"Unknown operation: {operation}"
                
        except Exception as e:
            return f"Error performing calculation: {str(e)}"


class MCPAIAssistant:
    """
    This is our main AI Assistant that uses MCP principles to coordinate between
    the AI model and various tools. Think of it as a smart manager who knows
    which employee (tool) to assign each task to.
    """
    
    def __init__(self):
        print("🤖 Initializing AI Assistant...")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize our tools (like hiring employees for different jobs)
        self.data_tool = DataAccessTool('customer_data.csv')
        self.calculator_tool = CalculatorTool()
        
        # Create a registry of available tools (like an employee directory)
        self.tools = {
            'query_customer_data': self.data_tool,
            'calculate': self.calculator_tool
        }
        
        print("✅ AI Assistant ready with the following tools:")
        print("   📊 Customer Data Access Tool")
        print("   🧮 Calculator Tool")
    
    def get_available_tools(self) -> List[Dict]:
        """
        Returns a list of all available tools for the AI to use
        This is like giving the AI a catalog of all available services
        """
        return [
            self.data_tool.get_tool_description(),
            self.calculator_tool.get_tool_description()
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Executes a specific tool with given parameters
        This is like the manager delegating a task to the right employee
        """
        print(f"🔧 Executing tool: {tool_name}")
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.tools[tool_name]
        return tool.execute(**kwargs)
    
    def process_query(self, user_query: str) -> str:
        """
        This is the main function that processes user queries.
        It's like a smart receptionist who understands what you need
        and directs you to the right department.
        """
        print(f"\n💭 Processing query: '{user_query}'")
        print("=" * 50)
        
        # Prepare the system message that explains the AI's role and available tools
        system_message = """You are an intelligent business assistant with access to customer data and calculation tools.

Available tools:
1. query_customer_data: Search customer database, get statistics, find specific customers
2. calculate: Perform mathematical calculations including discounts, percentages, basic math

When a user asks a question:
1. Determine if you need to use tools to answer the question
2. If tools are needed, respond with a JSON object specifying which tool to use and parameters
3. If no tools are needed, respond directly

For tool usage, respond in this exact format:
{"tool": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}

Examples:
- "Who is Alice?" → {"tool": "query_customer_data", "parameters": {"query_type": "find_customer", "filter_value": "Alice"}}
- "What's 15% of 1000?" → {"tool": "calculate", "parameters": {"operation": "percentage", "numbers": [1000], "percentage": 15}}
- "Show me electronics customers" → {"tool": "query_customer_data", "parameters": {"query_type": "category_stats", "filter_value": "Electronics"}}

Answer directly for general questions that don't require tools."""

        try:
            # Ask the AI to analyze the query and decide what to do
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1  # Low temperature for more consistent responses
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"🧠 AI Decision: {ai_response}")
            
            # Check if the AI wants to use a tool
            if ai_response.startswith('{') and ai_response.endswith('}'):
                try:
                    # Parse the tool request
                    tool_request = json.loads(ai_response)
                    tool_name = tool_request.get('tool')
                    parameters = tool_request.get('parameters', {})
                    
                    # Execute the tool
                    tool_result = self.execute_tool(tool_name, **parameters)
                    
                    # Ask the AI to format the final response
                    final_response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful business assistant. Format the following data into a clear, friendly response for the user."},
                            {"role": "user", "content": f"Original question: {user_query}\nTool result: {tool_result}\n\nPlease provide a clear, friendly response."}
                        ],
                        temperature=0.3
                    )
                    
                    return final_response.choices[0].message.content
                    
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as a direct response
                    return ai_response
            else:
                # Direct response from AI
                return ai_response
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


def main():
    """
    Main function that runs our AI assistant
    This is like opening the office for business!
    """
    print("🚀 Welcome to Your AI-Powered Business Assistant!")
    print("=" * 50)
    print("This assistant can help you with:")
    print("• Finding customer information")
    print("• Calculating discounts and percentages")
    print("• Analyzing sales data")
    print("• Answering business questions")
    print()
    
    # Initialize the assistant
    assistant = MCPAIAssistant()
    
    print("\n" + "=" * 50)
    print("🎯 Let's test the assistant with some example queries...")
    print("=" * 50)
    
    # Test queries to demonstrate functionality
    test_queries = [
        "Who is Alice Johnson?",
        "What are the sales statistics for Electronics?",
        "Calculate a 20% discount on $1250",
        "Who are the top 3 customers?",
        "What's the total revenue across all customers?",
        "Add 1250.50 and 890.25"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query}")
        print("-" * 30)
        result = assistant.process_query(query)
        print(f"📋 Answer: {result}")
        print()
    
    print("=" * 50)
    print("🎉 Interactive Mode - Ask your own questions!")
    print("(Type 'quit' to exit)")
    print("=" * 50)
    
    # Interactive mode
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for using the AI Assistant! Goodbye!")
                break
            
            if user_input:
                result = assistant.process_query(user_input)
                print(f"📋 Answer: {result}")
            
        except KeyboardInterrupt:
            print("\n👋 Thanks for using the AI Assistant! Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()


