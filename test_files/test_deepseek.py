# from openai import OpenAI

# client = OpenAI(api_key=X, base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)

from openai import OpenAI
# client = OpenAI()

# response = client.responses.create(
#     model="gpt-4o",
#     input="Write a one-sentence bedtime story about a unicorn."
# )

# print(response.output_text)

# import openai

client = OpenAI()
response = client.chat.completions.create(
    model="o4-mini-2025-04-16",
    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
    # max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

response_content = response.choices[0].message.content
print(response_content)