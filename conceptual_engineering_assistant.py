import yaml
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class Concept:
    """Represents a concept with a unique identifier, a label, and a definition."""
    
    def __init__(self, id, label, definition):
        """
        Initializes a concept with a unique identifier, a label, and a definition.
        
        Parameters:
            id: The unique identifier for the concept.
            label: The label or name of the concept.
            definition: The definition of the concept.
        """
        self.id = id
        self.label = label
        self.definition = definition

    def to_json(self):
        """
        Converts the concept to a JSON-like dictionary format.
        
        Returns:
            A dictionary with keys "id", "label", and "definition".
        """
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition           
        }

class Entity:
    """Represents an entity with a unique identifier, a label, and an optional description."""
    
    def __init__(self, id, label, description=""):
        """
        Initializes an entity with a unique identifier, a label, and an optional description.
        
        Parameters:
            id: The unique identifier for the entity.
            label: The label or name of the entity.
            description: An optional description of the entity.
        """
        self.id = id
        self.label = label
        self.description = description

    def to_json(self):
        """
        Converts the entity to a JSON-like dictionary format.
        
        Returns:
            A dictionary with keys "id", "label", and "description".
        """
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description           
        }

class ConceptualEngineeringAssistant:
    """Supports the process of conceptual engineering."""

    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes the assistant with a model name and temperature parameter.
        
        Parameters:
            model_name: The name of the model to be used (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self._classify_entity_chain = self._zero_shot_chain_of_thought('./chains/classify_entity.yaml')
        self._propose_counterexample_chain = self._zero_shot_chain_of_thought('./chains/propose_counterexample.yaml')
        self._refute_counterexample_chain = self._zero_shot_chain_of_thought('./chains/refute_counterexample.yaml')
        self._revise_definition_chain = self._zero_shot_chain_of_thought('./chains/revise_definition.yaml')

    def _zero_shot_chain_of_thought(self, file):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification Initializes the assistant with a model name and temperature parameter.
        
        Parameters:
            file: The name of the YAML file containing a specification of the CoT.
        """
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
        """
        Determines whether or not an entity is in the extension of the concept.
        
        Parameters:
            concept: The concept to be used for classification.
            entity: The entity to be classified.
        
        Returns:
            Classification results based on the concept's definition.
        """
        return self._classify_entity_chain(
            {
                "label": concept.label, 
                "definition": concept.definition, 
                "entity": entity.label,
                "description": entity.description
            }
        )
        
    def propose_counterexample(self, concept):
        """
        Provides a counterexample to the concept's definition.
        
        Parameters:
            concept: The concept to be examined.
        
        Returns:
            A proposed counterexample to the concept's definition.
        """
        return self._propose_counterexample_chain(
            {
                "label": concept.label, 
                "definition": concept.definition
            }
        )
    
    def refute_counterexample(self, counterexample):
        """
        Refutes the counterexample to the concept's definition.
        
        Parameters:
            counterexample: The counterexample to be refuted.
        
        Returns:
            A refutation of the counterexample.
        """
        return self._refute_counterexample_chain(counterexample)
    
    def revise_definition(self, counterexample):
        """
        Revises the concept's definition based on the counterexample.
        
        Parameters:
            counterexample: The counterexample to be used for revision.
        
        Returns:
            A revised definition of the concept.
        """
        return self._revise_definition_chain(counterexample)
