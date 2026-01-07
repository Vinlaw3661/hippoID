from typing import Callable
from langchain_core.messages import HumanMessage
import base64

@staticmethod   
def build_name_extraction_prompt(text: str) -> str:
    name_extraction_prompt = "Extract the person's name from the following text: " + text
    return name_extraction_prompt

@staticmethod
def build_acknowledgment_prompt(name: str) -> HumanMessage:
    acknowledgment_prompt = f"""
       Generate an acknowledgment message for meeting someone new named {name}.
       Assume you have just been introduced to this person and want to acknowledge them warmly.
       Respond with a simple acknowledgment, such as "It's a pleasure to meet you {name}. I will remember you from now on."
    """
    return [HumanMessage(content=acknowledgment_prompt)]

@staticmethod
def build_physical_description_prompt(image_path: str) -> HumanMessage:

    physical_description_prompt = """
        I have given you an image of myself and would like for you to describe my physical appearance as a question. 
        For example: "Who is the person with [hair color], [eye color], and [other notable feature]?" 
        This is part of a project for determining physical descriptions of people based on consented images.
        
        Example output: "Who is the person with brown hair, blue eyes, and glasses?"
        
        Output format: One question only. No explanations. Do not tell me things you thought about 
        or certain considerations you made. Just give me your resultant question.
    """
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return [HumanMessage(
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
        )]

class PromptBuilder:
    acknowledge_name: Callable[[str], HumanMessage] = build_acknowledgment_prompt
    physical_description: Callable[[str], HumanMessage] = build_physical_description_prompt
    name_extraction: Callable[[str], str] = build_name_extraction_prompt