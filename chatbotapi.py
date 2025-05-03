import requests
import json
import os
GOOGLE_API_KEY = 'AIzaSyBGOyCNLA6jKPsLnw6hc3l_Tcn0MrRzjH8'
def call_gemini_api(prompt, api_key=GOOGLE_API_KEY, model_name="gemini-2.0-flash"):
    """
    Calls the Gemini API to generate content based on a given prompt.

    Args:
        prompt: The text prompt to send to the Gemini API.
        api_key: (Optional) Your Gemini API key. If not provided, it will be read from the GOOGLE_API_KEY environment variable.
        model_name: (Optional) The name of the Gemini model to use. Defaults to "gemini-2.0-flash".

    Returns:
        The generated text from the API, or None if an error occurs.  Also returns the full API response for debugging.
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: Please set the GOOGLE_API_KEY environment variable or provide the api_key parameter.")
            return None, None  # Return None, None to indicate an error

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'contents': [
            {
                'parts': [
                    {'text': prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if 'candidates' in data and data['candidates']:
            output_text = data['candidates'][0]['content']['parts'][0]['text']
            with open('api_response.txt', 'w') as f:
                f.write(output_text)
            return output_text, data  # Return both text and full response
        else:
            print("No output generated.")
            return None, data  # Return None and full response
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None, None
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None, None
    except KeyError as e:
        print(f"Error accessing key in JSON response: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage:
    prompt = "Write a short poem about the city of Ho Chi Minh City."
    # You can either set the API key as an environment variable GOOGLE_API_KEY
    # or pass it directly to the function.
    # api_key = "YOUR_API_KEY"  # Replace with your actual API key
    generated_poem, full_response = call_gemini_api(prompt)

    if generated_poem:
        print("Generated Poem:")
        print(generated_poem)
    else:
        print("Failed to generate poem.")
    
    print("Full Response for debugging:")
    print(full_response) # print the full response.
