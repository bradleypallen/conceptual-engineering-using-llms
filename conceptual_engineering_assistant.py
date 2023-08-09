from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class Concept:

    def __init__(self, term, variable, definition):
        """Define a concept that provides an intentional definition for a given term."""
        self.term = term
        self.variable = variable
        self.definition = definition

class ConceptualEngineeringAssistant:

    def __init__(self, model_name="gpt-4", temperature=0.1):
        """Create an object that supports the process of conceptual engineering."""
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self._classify_entity_chain = self._classify_entity_chain()
        self._classify_entity_with_summary_chain = self._classify_entity_with_summary_chain()
        self._propose_counterexample_chain = self._propose_counterexample_chain()
        self._validate_counterexample_chain = self._validate_counterexample_chain()
        self._revise_concept_chain = self._revise_concept_chain()
    
    def _classify_entity_chain(self):
        """Generate a chain of thought for determining whether or not an entity is in the extension of a concept given its definition."""
        template_1 = "Definition: {variable} is a(n) {term} iff {definition}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["variable", "term", "definition", "entity"], 
            template=template_1
        )
        classification_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="in_extension")
        template_2 = "Definition: {variable} is a(n) {term} iff {definition}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer: {in_extension} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["variable", "term", "definition", "entity", "in_extension"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[classification_chain, explanation_chain],
            input_variables=["variable", "term", "definition", "entity"],
            output_variables=["entity", "in_extension", "rationale"]
        )
    
    def _classify_entity_with_summary_chain(self):
        """Generate a chain of thought for determining whether or not an entity is in the extension of a concept given its definition."""
        template_1 = "Definition: {variable} is a(n) {term} iff {definition}. {entity}: {summary}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["variable", "term", "definition", "entity", "summary"], 
            template=template_1
        )
        classification_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="in_extension")
        template_2 = "Definition: {variable} is a(n) {term} iff {definition}. {entity}: {summary}. Using the above definition, is {entity} a(n) {term}? Answer 'True', 'False', or 'Unknown'. Answer: {in_extension} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["variable", "term", "definition", "entity", "summary", "in_extension"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[classification_chain, explanation_chain],
            input_variables=["variable", "term", "definition", "entity", "summary"],
            output_variables=["entity", "in_extension", "rationale"]
        )
    
    def _propose_counterexample_chain(self):
        """Generate a chain of thought for proposing a counterexample to a concept's definition."""
        template_1 = "Definition: {variable} is a(n) {term} iff {definition}. Now imagine an opponent who challenges your definition and presents a potential counterexample of an entity that does not fit the definition but in the judgment of the opponent is in the extension of the concept. What is the name of that counterexample? Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["variable", "term", "definition"], 
            template=template_1
        )
        counterexample_proposal_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="counterexample")
        template_2 = "Definition: {variable} is a(n) {term} iff {definition}. Now imagine an opponent who challenges your definition and presents a potential counterexample of an entity that does not fit the definition but in the judgment of the opponent is in the extension of the concept. What is the name of that counterexample? Answer: {counterexample} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["variable", "term", "definition", "counterexample"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[counterexample_proposal_chain, explanation_chain],
            input_variables=["variable", "term", "definition"],
            output_variables=["counterexample", "rationale"]
        )
  
    def _validate_counterexample_chain(self):
        """Generate a chain of thought for arguing for or against the validity of a counterexample."""
        template_1 = "Definition: {variable} is a(n) {term} iff {definition}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Is this a valid counterexample? Answer 'True', 'False', or 'Unknown'. Answer:"
        prompt_1 = PromptTemplate(
            input_variables=["variable", "term", "definition", "counterexample"], 
            template=template_1
        )
        counterexample_validation_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="is_valid")
        template_2 = "Definition: {variable} is a(n) {term} iff {definition}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Is this a valid counterexample? Answer 'True', 'False', or 'Unknown'. Answer: {is_valid} Explain your reasoning. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["variable", "term", "definition", "counterexample", "is_valid"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[counterexample_validation_chain, explanation_chain],
            input_variables=["variable", "term", "definition", "counterexample"],
            output_variables=["is_valid", "rationale"]
        )

    def _revise_concept_chain(self):
        """Generate chain of thought for revising a concept based on a valid counterexample."""
        template_1 = "Definition: {variable} is a(n) {term} iff {definition}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Revise your definition to account for the counterexample. Revised definition:"
        prompt_1 = PromptTemplate(
            input_variables=["variable", "term", "definition", "counterexample"], 
            template=template_1
        )
        concept_revision_chain = LLMChain(llm=self.llm, prompt=prompt_1, output_key="revision")
        template_2 = "Definition: {variable} is a(n) {term} iff {definition}. Now imagine an opponent has challenged your definition by presenting {counterexample} as a counterexample. Revise your definition to account for the counterexample. Revised definition: {revision} Explain your reasoning as to why the revision accounts for the counterexample. Rationale:"
        prompt_2 = PromptTemplate(
            input_variables=["variable", "term", "definition", "counterexample", "revision"], 
            template=template_2,
        )
        explanation_chain = LLMChain(llm=self.llm, prompt=prompt_2, output_key="rationale")
        return SequentialChain(
            chains=[concept_revision_chain, explanation_chain],
            input_variables=["variable", "term", "definition", "counterexample"],
            output_variables=["revision", "rationale"]
        )

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
    
    def classify_entity_with_summary(self, concept, entity, summary):
        """Determine whether or not an entity is in the extension of the concept."""
        return self._classify_entity_with_summary_chain(
            {
                "variable": concept.variable, 
                "term": concept.term, 
                "definition": concept.definition, 
                "entity": entity,
                "summary": summary
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
                "counterexample": counterexample
            }
        )
    
    def revise_concept(self, concept, counterexample):
        """Revise the concept's definition based on the counterexample."""
        return self._revise_concept_chain(
            {
                "variable": concept.variable, 
                "term": concept.term, 
                "definition": concept.definition,
                "counterexample": counterexample
            }
        )

