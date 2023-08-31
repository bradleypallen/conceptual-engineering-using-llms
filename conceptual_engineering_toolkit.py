import yaml, requests, json, wikipedia, pycm, pandas as pd, numpy as np
from urllib.parse import urlparse, unquote
from datetime import datetime
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WikipediaLoader

class Concept:
    """Represents a concept with a unique identifier, a label, and a definition."""
    
    def __init__(self, id, label, definition, model_name="gpt-4", temperature=0.1):
        """
        Initializes a concept with a unique identifier, a label, and a definition.
        
        Parameters:
            id: The unique identifier for the concept.
            label: The label or name of the concept.
            definition: The definition of the concept.
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.id = id
        self.label = label
        self.definition = definition
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self._classify_chain = self._zero_shot_chain_of_thought('./chains/classify.yaml')
        self._propose_counterexample_chain = self._zero_shot_chain_of_thought('./chains/propose_counterexample.yaml')
        self._refute_counterexample_chain = self._zero_shot_chain_of_thought('./chains/refute_counterexample.yaml')
        self._revise_definition_chain = self._zero_shot_chain_of_thought('./chains/revise_definition.yaml')

    def to_json(self):
        """
        Converts the concept to a JSON-like dictionary format.
        
        Returns:
            A dictionary with keys "id", "label", and "definition".
        """
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition,
            "model_name": self.model_name,
            "temperature": self.temperature,           
        }
    
    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
        elif model_name in [
            "text-curie-001"
            ]:
            return OpenAI(model_name=model_name, temperature=temperature)
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "google/flan-t5-xxl",
            ]:
            return HuggingFaceHub(repo_id=model_name, model_kwargs={ "temperature": temperature })
        else:
            raise Exception(f'Model {model_name} not supported')

    def _zero_shot_chain_of_thought(self, file):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification.
        
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
    
    def classify(self, entity):
        """
        Determines whether or not an entity is in the extension of the concept.
        
        Parameters:
            entity: The entity to be classified.
        
        Returns:
            Classification results based on the concept's definition.
        """
        return self._classify_chain(
            {
                "label": self.label, 
                "definition": self.definition, 
                "entity": entity.label,
                "description": entity.description
            }
        )
        
    def propose_counterexample(self):
        """
        Provides a counterexample to the concept's definition.
        
        Returns:
            A proposed counterexample to the concept's definition.
        """
        return self._propose_counterexample_chain(
            {
                "label": self.label, 
                "definition": self.definition
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

class Benchmark:

    WIKIDATA_ENDPOINT = 'https://query.wikidata.org/sparql'

    def __init__(self, target_concept, positives_query_filename, negatives_query_filename, limit=1000) -> None:
        self.target_concept = target_concept
        self.limit = limit
        positives_limit = self.limit // 2
        negatives_limit = self.limit - positives_limit
        with open(positives_query_filename, 'r') as positives_query_file:
            self.positives_query = positives_query_file.read() + f'\nLIMIT {positives_limit}'
        with open(negatives_query_filename, 'r') as negatives_query_file:
            self.negatives_query = negatives_query_file.read() + f'\nLIMIT {negatives_limit}'
 
    def retrieve(self):
        pos_response = requests.get(self.WIKIDATA_ENDPOINT, params={'query' : self.positives_query, 'format' : 'json'})
        pos_response.raise_for_status()
        self.positives = [ { k: v["value"] for k, v in result.items() } for result in json.loads(pos_response.text)["results"]["bindings"] ]
        for item in self.positives:
            item["@id"] = item.pop("item")
            item["actual"] = 'positive'
            path = urlparse(item.pop("article")).path
            page_title = unquote(path.removeprefix("/wiki/"))
            try:
                item["description"] = wikipedia.summary(page_title, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                item["description"] = "Disambiguation error when retrieving summary"
        neg_response = requests.get(self.WIKIDATA_ENDPOINT, params={'query' : self.negatives_query, 'format' : 'json'})
        neg_response.raise_for_status()
        self.negatives = [ { k: v["value"] for k, v in result.items() } for result in json.loads(neg_response.text)["results"]["bindings"] ]
        for item in self.negatives:
            item["@id"] = item.pop("item")
            item["actual"] = 'negative'
            path = urlparse(item.pop("article")).path
            page_title = unquote(path.removeprefix("/wiki/"))
            try:
                item["description"] = wikipedia.summary(page_title, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                item["description"] = "Disambiguation error when retrieving summary"

    def save(self):
        benchmark = {
            "created": datetime.now().isoformat(),
            "target_concept": self.target_concept,
            "positives": {
                "query": self.positives_query,
                "data": self.positives,
            },
            "negatives": {
                "query": self.negatives_query,
                "data": self.negatives,
            }
        }
        filename = f'benchmarks/{self.target_concept}/data.json'
        with open(filename, 'w+') as file:
            json.dump(benchmark, file)
        return filename

class Experiment:

    def __init__(self, benchmark_file, concept_file, model_name, temperature=0.1, use_description=True) -> None:
        self.benchmark = json.load(open(benchmark_file, 'r'))
        concept_json = yaml.safe_load(open(concept_file, 'r'))
        self.concept = Concept(concept_json["id"], concept_json["label"], concept_json["definition"], model_name, temperature)
        self.use_description = use_description

    def sample(self, n=40):
        positives = self.benchmark["positives"]["data"]
        negatives = self.benchmark["negatives"]["data"]
        n_positives = min(len(positives), n // 2)
        n_negatives = min(len(negatives), n - n_positives)
        return np.concatenate((
            np.random.choice(positives, size=n_positives, replace=False), 
            np.random.choice(negatives, size=n_negatives, replace=False) 
        ))

    def run(self, sample):
        if self.use_description:
            predictions = [ self.concept.classify(Entity(entity["@id"], entity["name"], entity["description"])) for entity in sample ]
        else:
            predictions = [ self.concept.classify(Entity(entity["@id"], entity["name"])) for entity in sample ]
        df_pred = pd.DataFrame(predictions, columns = [ 'entity' , 'predicted', 'rationale' ])
        df_pred["predicted"] = df_pred["predicted"].str.lower()
        df_samp = pd.DataFrame.from_records(sample)
        self.results = pd.concat([df_samp[[ "name", "@id", "description", "actual" ]], df_pred[[ "predicted", "rationale" ]]], axis=1)

    def save(self):
        cm = pycm.ConfusionMatrix(
            self.results["actual"].tolist(), 
            self.results["predicted"].tolist(), 
            digit=2, 
            classes=[ 'positive', 'negative' ]
        )
        run = {
            "created": datetime.now().isoformat(),
            "concept": self.concept.to_json(),
            "data": self.results.to_dict('records'),
            "confusion_matrix": cm.matrix,
        }
        filename = f'experiments/{run["concept"]["model_name"]}_{run["concept"]["id"]}_{run["created"]}.json'
        json.dump(run, open(filename, 'w'))
        return filename
