from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class Concept:

    def __init__(self, id, term, definition):
        """Define a concept that provides an intentional definition for a given term."""
        self.id = id
        self.term = term
        self.definition = definition

class ConceptualEngineeringAssistant:

    def __init__(self, model_name="gpt-4", temperature=0.1):
        """Create an object that supports the process of conceptual engineering."""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self._classify_entity_chain = self._classify_entity_chain()
        self._propose_counterexample_chain = self._propose_counterexample_chain()
        self._validate_counterexample_chain = self._validate_counterexample_chain()
        self._revise_concept_chain = self._revise_concept_chain()
        
    def _classify_entity_chain(self):
        """Generate a chain of thought for determining whether or not an entity is in the extension of a concept given its definition."""
        template_1 = "Term: {term} Definition: {definition}. Entity: {entity} Description: {description}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["term", "definition", "entity", "description"], 
            template=template_1
        )
        classification_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="in_extension")
        template_2 = "Term: {term} Definition: {definition}. Entity: {entity} Description: {description}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer: {in_extension} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["term", "definition", "entity", "description", "in_extension"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[classification_chain, explanation_chain],
            input_variables=["term", "definition", "entity", "description"],
            output_variables=["entity", "in_extension", "rationale"]
        )
    
    def _propose_counterexample_chain(self):
        """Generate a chain of thought for proposing a counterexample to a concept's definition."""
        template_1 = "Term: {term} Definition: {definition}. Now imagine an opponent who challenges your definition and presents a potential counterexample of an entity that does not fit the definition but in the judgment of the opponent is in the extension of the concept. What is the name of that counterexample? Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["term", "definition"], 
            template=template_1
        )
        counterexample_proposal_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="counterexample")
        template_2 = "Term: {term} Definition: {definition}. Now imagine an opponent who challenges your definition and presents a potential counterexample of an entity that does not fit the definition but in the judgment of the opponent is in the extension of the concept. What is the name of that counterexample? Answer: {counterexample} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["term", "definition", "counterexample"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[counterexample_proposal_chain, explanation_chain],
            input_variables=["term", "definition"],
            output_variables=["counterexample", "rationale"]
        )
  
    def _validate_counterexample_chain(self):
        """Generate a chain of thought for arguing for or against the validity of a counterexample."""
        template_1 = "Term: {term} Definition: {definition}. Entity: {counterexample} Description: {description}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Is this a valid counterexample? Answer 'True', 'False', or 'Unknown'. Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["term", "definition", "counterexample", "description"], 
            template=template_1
        )
        counterexample_validation_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="is_valid")
        template_2 = "Term: {term} Definition: {definition}. Entity: {counterexample} Description: {description}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Is this a valid counterexample? Answer 'True', 'False', or 'Unknown'. Answer: {is_valid} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["term", "definition", "counterexample", "description", "is_valid"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[counterexample_validation_chain, explanation_chain],
            input_variables=["term", "definition", "counterexample", "description"],
            output_variables=["is_valid", "rationale"]
        )

    def _revise_concept_chain(self):
        """Generate chain of thought for revising a concept based on a valid counterexample."""
        template_1 = "Term: {term} Definition: {definition}. Entity: {counterexample} Description: {description}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Revise your definition to account for the counterexample. Revised definition:"
        prompt_1 = PromptTemplate(
            input_variables=["term", "definition", "counterexample", "description"], 
            template=template_1
        )
        concept_revision_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="revision")
        template_2 = "Term: {term} Definition: {definition}. Entity: {counterexample} Description: {description}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Revise your definition to account for the counterexample. Revised definition: {revision} Explain your reasoning as to why the revision accounts for the counterexample. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["term", "definition", "counterexample", "description", "revision"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[concept_revision_chain, explanation_chain],
            input_variables=["term", "definition", "counterexample", "description"],
            output_variables=["revision", "rationale"]
        )

    def classify_entity(self, concept, entity, description="No description available"):
        """Determine whether or not an entity is in the extension of the concept."""
        return self._classify_entity_chain(
            {
                "term": concept.term, 
                "definition": concept.definition, 
                "entity": entity,
                "description": description
            }
        )
    
    def classify_entity_with_description(self, concept, entity, description):
        """Determine whether or not an entity is in the extension of the concept."""
        return self._classify_entity_with_description_chain(
            {
                "term": concept.term, 
                "definition": concept.definition, 
                "entity": entity,
                "description": description
            }
        )
    
    def propose_counterexample(self, concept):
        """Provide a counterexample to the concept's definition."""
        return self._propose_counterexample_chain(
            {
                "term": concept.term, 
                "definition": concept.definition
            }
        )
    
    def validate_counterexample(self, concept, counterexample, description="No description available"):
        """Validate the counterexample to the concept's definition."""
        return self._validate_counterexample_chain(
            {
                "term": concept.term, 
                "definition": concept.definition,
                "counterexample": counterexample,
                "description": description
            }
        )
    
    def revise_concept(self, concept, counterexample, description="No description available"):
        """Revise the concept's definition based on the counterexample."""
        return self._revise_concept_chain(
            {
                "term": concept.term, 
                "definition": concept.definition,
                "counterexample": counterexample,
                "description": description
            }
        )

