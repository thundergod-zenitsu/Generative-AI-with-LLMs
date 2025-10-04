import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key-here'

def get_gpt_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Specify the model
            prompt=prompt,
            max_tokens=150,  # Limit the response length
            n=1,  # Number of responses to generate
            stop=None,  # Define stopping criteria
            temperature=0.7  # Control the creativity of the output
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

# Streamlit app
st.title('GPT-3 Code Assistant')
st.write('Enter a prompt and get a response from GPT-3')

# Text input for the prompt
user_prompt = st.text_area('Enter your prompt here (e.g., "Write a Python function to reverse a string"):')

# Button to submit the prompt
if st.button('Get Response'):
    if user_prompt:
        with st.spinner('Generating response...'):
            response = get_gpt_response(user_prompt)
        
        # Determine if the response contains code
        if "```" in response:
            # Split the response to extract the code part
            parts = response.split("```")
            # The code part is usually within the triple backticks
            code = parts[1] if len(parts) > 1 else response
            st.write('### Response')
            st.code(code, language='python')  # You can specify other languages as needed
        else:
            st.write('### Response')
            st.write(response)
    else:
        st.write('Please enter a prompt.')
