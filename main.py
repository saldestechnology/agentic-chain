from transformers import pipeline
from pydantic import BaseModel, ConfigDict, SerializeAsAny
from typing import Any, Callable

from pprint import pprint

class Runnable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def invoke(self, data: Any) -> Any:
        raise NotImplementedError("I am not implemented! :o")

    def __or__(self, other: Any) -> Any:
        if isinstance(other, Runnable):
            return RunnableSequence(
                first=self,
                second=other
            )
        if callable(other):
            return RunnableSequence(
                first=self,
                second=RunnableLambda(func=other)
            )
        return NotImplemented
    
    def __ror__(self, other: Any) -> Any:
        if callable(other):
            return RunnableSequence(
                first=RunnableLambda(func=other),
                second=self,
            )
        

class RunnableLambda(Runnable):
    func:  Callable[[Any], Any]
    
    def invoke(self, data: Any) -> Callable[[Any], Any]:
        return self.func(data)

class RunnableSequence(Runnable):
    first: SerializeAsAny[Runnable]
    second: SerializeAsAny[Runnable]

    def invoke(self, data: Any) -> Any:
        return self.second.invoke(self.first.invoke(data))

# --- DTO ---

class TicketInput(BaseModel):
    customer_id: int
    message: str
    
class ProcessedTicket(BaseModel):
    customer_id: int
    sentiment: str
    urgency: str
    summary: str
    
# --- Functionality for our pipeline ---

class SentimentAnalyser(Runnable):
    name: str = "sentiment"
    def invoke(self, ticket: TicketInput) -> dict:
        msg_lower = ticket.message.lower()

        # Simulated NLP sentiment
        sentiment = "negative" if "broken" in msg_lower or "angry" in msg_lower else "neutral"
        urgency = "high" if "broken" in msg_lower or "urgent" in msg_lower else "low"

        return {
            "customer_id": ticket.customer_id,
            "sentiment": sentiment,
            "urgency": urgency,
            "summary": ticket.message[:40] + "..."
        }

class TicketParser(Runnable):
    name: str = "ticket_parser"
    def invoke(self, raw_dict: dict) -> ProcessedTicket:
        return ProcessedTicket(**raw_dict)

def route_ticket(ticket: ProcessedTicket) -> dict:
    destination = "engineering_team" if ticket.urgency == "high" else "general_support"
    return {
        "status": "routed",
        "assigned_to": destination,
        "ticket_details": ticket.model_dump()
    }

ticket_pipeline = SentimentAnalyser() | TicketParser() | route_ticket

incoming_ticket = TicketInput(
    customer_id=1337,
    message="The payment portal is broken! Urgent fix is needed ASAP!"
)

final_output = ticket_pipeline.invoke(incoming_ticket)

print("--- PIPELINE EXECUTED SUCCESSFULLY ---")
pprint(final_output)

print("--- INSIGHT ---")
pprint(ticket_pipeline.model_dump(exclude_none=True))



# --- Class based ---

class SmolLM:
    def __init__(self, model_name="HuggingFaceTB/SmolLM-135M-Instruct"):
        print("Loading {model_name} into memory (this may take a while)...")
        self.pipe = pipeline("text-generation", model_name)
        print("Model loaded successfully!")
    
    def invoke(self, prompt: str):
        messages = [{ "role": "user", "content": prompt }]
        output = self.pipe(messages, max_new_tokens=150)
        return output[0]['generated_text'][-1]['content']

class PromptTemplate:
    def __init__(self, template_str: str):
        self.template_str = template_str
    
    def format(self, **kwargs):
        return self.template_str.format(**kwargs)
        
    
    def __or__(self, other):
        if isinstance(other, SmolLM):
            return LLMChain(
                prompt_template=self, 
                llm=other
            )
        raise TypeError("It's not an instance of SmolLM!")

class LLMChain:
    def __init__(
        self,
        prompt_template: PromptTemplate,
        llm: SmolLM
    ):
        self.prompt_template = prompt_template
        self.llm = llm
    
    def invoke(self, **kwargs):
        formatted_prompt = self.prompt_template.format(**kwargs)
        return self.llm.invoke(formatted_prompt)

# llm = SmolLM()

# recipe_prompt = PromptTemplate(
    
#     template_str="Give me a quick 2-step recipe for a {dish} using only {ingredient_count} ingredients."
# )

# recipe_chain = recipe_prompt | llm

# result = recipe_chain.invoke(dish="pie", ingredient_count="four")
# pprint(result)

