"""
OpenAI Tool Integration Demo
Shows how AI models can use external tools for real-time data
"""

import requests
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()


def get_weather(city):
    """Weather tool for AI to access current weather"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data["current_condition"][0]

        return {
            "city": city,
            "temperature": f"{current['temp_C']}°C",
            "description": current['weatherDesc'][0]['value'],
            "humidity": f"{current['humidity']}%"
        }
    except Exception as e:
        return {"error": f"Weather unavailable: {str(e)}"}


def get_stock_price(symbol):
    """Stock price tool for AI"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            result = data['chart']['result'][0]['meta']
            price = result['regularMarketPrice']
            prev_close = result['previousClose']
            change = price - prev_close
            change_percent = (change / prev_close) * 100

            return {
                'symbol': symbol,
                'price': f"${price:.2f}",
                'change': f"${change:.2f}",
                'change_percent': f"{change_percent:.2f}%"
            }
    except Exception as e:
        return {"error": f"Stock data unavailable: {str(e)}"}


# Define tools for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"}
                },
                "required": ["symbol"]
            }
        }
    }
]


def chat_with_tools(user_message):
    """Send message to OpenAI with tool access"""
    print(f"🗨️  User: {user_message}")

    # Call OpenAI with tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Check if AI wants to use tools
    if message.tool_calls:
        print("🔧 AI is using tools...")

        # Execute tool calls
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "get_weather":
                result = get_weather(arguments["city"])
            elif function_name == "get_stock_price":
                result = get_stock_price(arguments["symbol"])

            print(f"   Tool result: {result}")

            # Send tool result back to AI
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_message},
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    }
                ]
            )

            print(f"🤖 AI: {final_response.choices[0].message.content}")
    else:
        print(f"🤖 AI: {message.content}")


def main():
    """Demo OpenAI with tools"""
    print("🚀 OpenAI Tool Integration Demo")
    print("=" * 40)

    # Demo queries
    queries = [
        "What's the weather in New York?",
        "How is Tesla stock doing?"
    ]

    for query in queries:
        print()
        chat_with_tools(query)
        print("-" * 40)


if __name__ == "__main__":
    main()