# kg_agents.py
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from typing import TYPE_CHECKING, Optional, Union, Dict, Any
import re

if TYPE_CHECKING:
    from unstructured.documents.elements import Element

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.prompts import TextPrompt
from camel.storages.graph_storages.graph_element import (
    GraphElement,
    Node,
    Relationship,
)
from camel.types import RoleType

try:
    import os
    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import track_agent
    else:
        raise ImportError
except (ImportError, AttributeError):
    from camel.utils import track_agent

text_prompt = """
You are tasked with extracting entities (nodes) and relationships from historical spine science texts and traditional medicine documents, then structuring them into Node and Relationship objects. Whatever the language of the documents, your extracted nodes and relationships should be translated in English. Here's the outline of what you need to do:

Content Extraction:
You should be able to process input content and identify entities mentioned within it.
Focus on entities related to historical spine treatments, observations, and medical concepts.

Node Extraction:
For each identified entity, create a Node object.
Each Node object should have a unique identifier (id) and a type (type).
Node types should be one of the following:
- ClinicalObservation (signs, symptoms, disease presentations)
- TherapeuticOutcome (treatment responses, recovery patterns)
- ContextualFactor (environmental, behavioral, constitutional factors)
- MechanisticConcept (traditional explanatory models, processes)
- TherapeuticApproach (interventions, remedies, methods)
- SourceText (reference to original documents or authors)
Additional properties associated with the node can also be extracted and stored.

Relationship Extraction:
Identify relationships between extracted entities in the content.
For each relationship, create a Relationship object.
A Relationship object should have a subject (subj) and an object (obj) which are Node objects.
Each relationship should have a type (type) from the following options:
- co_occurs_with (between related clinical observations)
- preceded_by/followed_by (temporal relationships)
- modified_by (how contexts affect observations)
- responds_to (observation responses to treatments)
- associated_with (contextual associations with observations)
- results_in (effects produced by treatments)
- described_in (attribution to source texts)
- contradicts/corroborates (consistency relationships)
Add any relevant qualifiers as additional properties if applicable.

Output Formatting:
The extracted nodes and relationships should be formatted as instances of the provided Node and Relationship classes.
Ensure that the extracted data adheres to the structure defined by the classes.
Output the structured data in a format that can be easily validated against the provided code.
Do not wrap the output in lists or dictionaries, provide the Node and Relationship with unique identifiers.
Strictly follow the format provided in the example output, do not add any additional information.

Instructions for you:
Read the provided historical spine science content thoroughly.
Identify distinct entities mentioned in the content and categorize them using the specified node types.
Determine relationships between these entities using the relationship types provided.
Provide the extracted nodes and relationships in the specified format below.

Example for you:
Example Content:
"Ollivier describes cases of paralysis linked to spinal blood congestions, where an accumulation of blood in the spinal veins leads to symptoms like incomplete paralysis without intellectual impairment. He notes that these congestions often resolve spontaneously."

Expected Output:
Nodes:
Node(id='paralysis_spinal_blood_congestion', type='ClinicalObservation')
Node(id='incomplete_paralysis', type='ClinicalObservation')
Node(id='preserved_intellect', type='ClinicalObservation')
Node(id='blood_accumulation_spinal_veins', type='MechanisticConcept')
Node(id='spontaneous_resolution', type='TherapeuticOutcome')
Node(id='Ollivier', type='SourceText')

Relationships:
Relationship(subj=Node(id='blood_accumulation_spinal_veins', type='MechanisticConcept'), obj=Node(id='paralysis_spinal_blood_congestion', type='ClinicalObservation'), type='associated_with')
Relationship(subj=Node(id='paralysis_spinal_blood_congestion', type='ClinicalObservation'), obj=Node(id='incomplete_paralysis', type='ClinicalObservation'), type='co_occurs_with')
Relationship(subj=Node(id='paralysis_spinal_blood_congestion', type='ClinicalObservation'), obj=Node(id='preserved_intellect', type='ClinicalObservation'), type='co_occurs_with')
Relationship(subj=Node(id='paralysis_spinal_blood_congestion', type='ClinicalObservation'), obj=Node(id='spontaneous_resolution', type='TherapeuticOutcome'), type='results_in')
Relationship(subj=Node(id='paralysis_spinal_blood_congestion', type='ClinicalObservation'), obj=Node(id='Ollivier', type='SourceText'), type='described_in')

===== TASK =====
Please extracts nodes and relationships from the given content and structure them 
into Node and Relationship objects. 

{task}
"""

@track_agent(name="KnowledgeGraphAgent")
class KnowledgeGraphAgent(ChatAgent):
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
    ) -> None:
        system_message = BaseMessage(
            role_name="Graphify",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="Your mission is to transform unstructured content "
            "into structured graph data. Extract nodes and relationships with "
            "precision, and let the connections unfold. Your graphs will "
            "illuminate the hidden connections within the chaos of "
            "information.",
        )
        super().__init__(system_message, model=model)

    def run(
        self,
        element: "Element",
        parse_graph_elements: bool = False,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, GraphElement]:
        self.reset()
        self.element = element
        final_prompt = prompt if prompt is not None else text_prompt
        knowledge_graph_prompt = TextPrompt(final_prompt)
        knowledge_graph_generation = knowledge_graph_prompt.format(
            task=str(element))

        response = self.step(input_message=knowledge_graph_generation)
        content = response.msg.content

        if parse_graph_elements:
            content = self._parse_graph_elements(content, metadata)

        return content

    def _validate_node(self, node: Node) -> bool:
        return (isinstance(node, Node) and isinstance(node.id, (str, int))
                and isinstance(node.type, str))

    def _validate_relationship(self, relationship: Relationship) -> bool:
        return (isinstance(relationship, Relationship)
                and self._validate_node(relationship.subj)
                and self._validate_node(relationship.obj)
                and isinstance(relationship.type, str))

    def _parse_graph_elements(self, input_string: str, metadata: Optional[Dict[str, Any]] = None) -> GraphElement:
        node_pattern = r"Node\(id='(.*?)', type='(.*?)'\)"
        rel_pattern = (r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), "
                       r"obj=Node\(id='(.*?)', type='(.*?)'\), "
                       r"type='(.*?)'(?:, timestamp='(.*?)')?\)")

        nodes = {}
        relationships = []

        # Prepare Metadata
        metadata_properties = {'source': 'agent_created'}
        if metadata:
            metadata_properties.update(metadata)

        # Extract nodes
        for match in re.finditer(node_pattern, input_string):
            id, type = match.groups()
            
            # --- FIXED PROPERTIES ---
            # 1. 'name' is essential for Visualization captions 
            properties = {'name': id}
            
            # 2. Add metadata (Title, doc_id, filename)
            properties.update(metadata_properties)
            
            # Note: We removed 'entity_id' to avoid duplicates.
            # 'id' is automatically added by the database storage layer.
            
            if id not in nodes:
                node = Node(id=id, type=type, properties=properties)
                if self._validate_node(node):
                    nodes[id] = node

        # Extract relationships
        for match in re.finditer(rel_pattern, input_string):
            groups = match.groups()
            if len(groups) == 6:
                subj_id, subj_type, obj_id, obj_type, rel_type, timestamp = groups
            else:
                subj_id, subj_type, obj_id, obj_type, rel_type = groups
                timestamp = None
            
            properties = {'source': 'agent_created'}
            if metadata:
                properties.update(metadata)
            
            if subj_id in nodes and obj_id in nodes:
                subj = nodes[subj_id]
                obj = nodes[obj_id]
                relationship = Relationship(
                    subj=subj,
                    obj=obj,
                    type=rel_type,
                    timestamp=timestamp,
                    properties=properties,
                )
                if self._validate_relationship(relationship):
                    relationships.append(relationship)

        return GraphElement(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=self.element,
        )