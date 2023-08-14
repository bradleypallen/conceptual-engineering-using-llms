import yaml
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class Concept:

    def __init__(self, id, label, definition):
        """Define a concept."""
        self.id = id
        self.label = label
        self.definition = definition

    def to_json(self):
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition           
        }
    
class Entity:

    def __init__(self, id, label, description=""):
        self.id = id
        self.label = label
        self.description = description

    def to_json(self):
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition           
        }


class ConceptualEngineeringAssistant:

    def __init__(self, model_name="gpt-4", temperature=0.1):
        """Create an object that supports the process of conceptual engineering."""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self._classify_entity_chain = self._zero_shot_chain_of_thought('./chains/classify_entity.yaml')
        self._propose_counterexample_chain = self._zero_shot_chain_of_thought('./chains/propose_counterexample.yaml')
        self._refute_counterexample_chain = self._zero_shot_chain_of_thought('./chains/refute_counterexample.yaml')
        self._revise_definition_chain = self._zero_shot_chain_of_thought('./chains/revise_definition.yaml')

    def _zero_shot_chain_of_thought(self, file):
        chain_specification = yaml.safe_load(open(file, 'r'))
        template_1 = chain_specification["rationale_generation"]
        chain_1 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=template_1["input_variables"], 
                template=template_1["template"]
            ), 
            output_key=template_1["output_key"]
        )
        template_2 = chain_specification["answer_generation"]
        chain_2 = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=template_2["input_variables"], 
                template=template_2["template"]
            ), 
            output_key=template_2["output_key"]
        )
        return SequentialChain(
            chains=[chain_1, chain_2],
            input_variables=template_1["input_variables"],
            output_variables=chain_specification["output_variables"]
        )
    
    def classify_entity(self, concept, entity):
        """Determine whether or not an entity is in the extension of the concept."""
        return self._classify_entity_chain(
            {
                "label": concept.label, 
                "definition": concept.definition, 
                "entity": entity.label,
                "description": entity.description
            }
        )
        
    def propose_counterexample(self, concept):
        """Provide a counterexample to the concept's definition."""
        return self._propose_counterexample_chain(
            {
                "label": concept.label, 
                "definition": concept.definition
            }
        )
    
    def refute_counterexample(self, counterexample):
        """Refute the counterexample to the concept's definition."""
        return self._refute_counterexample_chain(counterexample)
    
    def revise_definition(self, counterexample):
        """Revise the concept's definition based on the counterexample."""
        return self._revise_definition_chain(counterexample)
