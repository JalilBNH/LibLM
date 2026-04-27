from openai import OpenAI
import requests

def unload_models(model_name: str, host: str = "http://localhost:11434"):
    """_summary_

    Args:
        model_name (str): _description_
        host (_type_, optional): _description_. Defaults to "http://localhost:11434".
    """
    
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model_name,
            "keep_alive": 0
        }
    )
    response.raise_for_status()
    
def main():
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama'
    )
    
    responses_result = client.responses.create(
        model='mistral',
        instructions='Answer to me like a singer trying to rhyme',
        input='How are you man ?',
        
    )
    print(responses_result.output_text)
    
    unload_models('mistral')

if __name__ == "__main__":
    main()
