from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class Concept:

    def __init__(self, term, variable, definition):
        self.term = term
        self.variable = variable
        self.definition = definition

class ConceptualEngineeringAssistant:

    def __init__(self, model_name="gpt-4", temperature=0.1):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self._classify_entity_chain = self._classify_entity_chain()
        self._propose_counterexample_chain = self._propose_counterexample_chain()
        self._revise_concept_chain = self._revise_concept_chain()
        self._validate_counterexample_chain = self._validate_counterexample_chain()
     
    def _classify_entity_chain(self):
        template_1 = "Definition: {variable} is a(n) {term} iff {definition}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["variable", "term", "definition", "entity"], 
            template=template_1
        )
        template_2 = "Definition: {variable} is a(n) {term} iff {definition}. Using the above definition, is {entity} a(n) {term}? Answer 'True' or 'False'. Answer: {in_extension} Explain your reasoning. Explanation:"
        prompt_2 = PromptTemplate(
            input_variables=["variable", "term", "definition", "entity", "in_extension"], 
            template=template_2,
        )
        in_extension_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="in_extension")
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[in_extension_chain, explanation_chain],
            input_variables=["variable", "term", "definition", "entity"],
            output_variables=["entity", "in_extension", "rationale"]
        )
    
    def _propose_counterexample_chain(self):
        return
  
    def _revise_concept_chain(self):
        return

    def _validate_counterexample_chain(self):
        return

    def classify_entity(self, concept, entity):
        """Determine whether or not an entity is in the extension of the concept."""
        return self._classify_entity_chain(
            {
                "variable": concept.variable, 
                "term": concept.term, 
                "definition": concept.definition, 
                "entity": entity
            }
        )
    
    def propose_counterexample(self, concept):
        """Provide a counterexample to the concept's definition."""
        return self._propose_counterexample_chain(
            {
                "variable": concept.variable, 
                "term": concept.term, 
                "definition": concept.definition
            }
        )
    
    def validate_counterexample(self, concept, counterexample):
        """Validate the counterexample to the concept's definition."""
        return self._validate_counterexample_chain(
            {
                "variable": concept.variable, 
                "term": concept.term, 
                "definition": concept.definition,
                "entity": counterexample
            }
        )
    
    def revise_concept(self, concept, counterexample):
        """Revise the concept's definition based on the counterexample."""
        return self._revise_concept_chain(
            {
                "variable": concept.variable, 
                "term": concept.term, 
                "definition": concept.definition,
                "entity": counterexample
            }
        )

