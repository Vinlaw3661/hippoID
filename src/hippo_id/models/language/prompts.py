from typing import Callable
from langchain_core.messages import HumanMessage
import base64

def build_ask_person_name_prompt() -> HumanMessage:
    ask_person_name_prompt = """
        Please provide the name of the person in the image.
        Respond with the name in the format: "Person Name: [Name]".
    """
    return HumanMessage(content=ask_person_name_prompt)

def build_acknowledgment_prompt(name: str) -> HumanMessage:
    acknowledgment_prompt = f"""
        Acknowledge the person named {name} in the provided text.
        Respond with a simple acknowledgment, such as "I acknowledge {name}."
    """
    return HumanMessage(content=acknowledgment_prompt)

def build_physical_description_prompt(image_path: str) -> HumanMessage:

    physical_description_prompt = """
        Describe what is in the image in a way that a person could understand. Do not include a description of the black background. 
        Frame your response as a question asking who they are. Here are some examples:
            
        Who is the person with the black hair and hazel eyes?

        Who is the person with the blonde hair, blue eyes, and green hoop earrings?

        NOTE: Respond only with the question and nothing else. Do not add any additional text to your response.
    """
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": physical_description_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                    }
                    }
                ]
        )

class PromptBuilder:
    ask_name: Callable[[], HumanMessage] = build_ask_person_name_prompt
    acknowledge_name: Callable[[str], HumanMessage] = build_acknowledgment_prompt
    physical_description: Callable[[str], HumanMessage] = build_physical_description_prompt