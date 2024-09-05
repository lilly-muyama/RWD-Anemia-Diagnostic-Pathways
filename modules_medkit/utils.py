import pandas as pd
import sys
sys.path.append('..')
from modules_medkit.constants import constants
from modules_medkit import utils
from modules_medkit.modified_section_tokenizer import ModifiedSectionTokenizer
from modules_medkit.final_entity_finder import FinalEntityFinder
from pathlib import Path
from medkit.core.text import TextDocument
from medkit.text.segmentation import SectionTokenizer, SyntagmaTokenizer, SentenceTokenizer
from medkit.text.context import FamilyDetector, NegationDetector, HypothesisDetector
from medkit.core import PipelineStep, Pipeline
from iamsystem import Matcher
from iamsystem import ESpellWiseAlgo
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule
from medkit.text.ner.iamsystem_matcher import IAMSystemMatcher
from medkit.text.ner.iamsystem_matcher import MedkitKeyword

family_rules = FamilyDetector.load_rules("../rules/family_testing_rules.yml")
negation_rules = NegationDetector.load_rules("../rules/negation_testing_rules.yml")
hypothesis_rules = HypothesisDetector.load_rules("../rules/hypothesis_testing_rules.yml")

def define_pipeline(section_dict):
    modified_sect_tokenizer = ModifiedSectionTokenizer(section_dict=section_dict, output_label="section")
    sentence_tokenizer = SentenceTokenizer(output_label="sentence", keep_punct=True, split_on_newlines=True, attrs_to_copy=["section"])
    syntagma_tokenizer = SyntagmaTokenizer(output_label="syntagma", attrs_to_copy=["section"])

    family_detector = FamilyDetector(rules=family_rules, output_label="family")
    negation_detector = NegationDetector(rules=negation_rules, output_label="negation")
    hypothesis_detector = HypothesisDetector(rules=hypothesis_rules, output_label="hypothesis")

    entity_finder = FinalEntityFinder()

    keywords_list=[]
    terms = pd.read_csv(constants.TERMS_FILEPATH)
    i = 0
    for term in terms[constants.TERMS_COLUMN]:
        keywords_list.append(MedkitKeyword(label=term, kb_id=str(i), kb_name="Anemia", ent_label='DISORDERS'))
        i = i+1

    matcher = Matcher.build(
        keywords=keywords_list,
        spellwise=[dict(measure=ESpellWiseAlgo.LEVENSHTEIN,max_distance=1,min_nb_char=10,)],
        stopwords=["et"],
        w=2,
    )
    iam_matcher = IAMSystemMatcher(matcher = matcher, attrs_to_copy = ["section", "family", "hypothesis", "negation"])

    pipeline_steps = [
        PipelineStep(modified_sect_tokenizer, input_keys=["text"], output_keys=["sections"]),
        PipelineStep(sentence_tokenizer, input_keys=["sections"], output_keys=["sentences"]),
        PipelineStep(syntagma_tokenizer, input_keys=["sentences"], output_keys=["syntagmas"]),  
        PipelineStep(family_detector, input_keys=["syntagmas"], output_keys=[]),  
        PipelineStep(negation_detector, input_keys=["syntagmas"], output_keys=[]),  
        PipelineStep(hypothesis_detector, input_keys=["syntagmas"], output_keys=[]),  
        PipelineStep(iam_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
        PipelineStep(entity_finder, input_keys=["entities"], output_keys=["final_entities"])
    ]
    pipeline = Pipeline(pipeline_steps, input_keys=["text"], output_keys=["final_entities"])
    return pipeline

def visualize_entities(entities, note):
    options_displacy = dict(colors={"anemia": "#7CFC00"})
    displacy_data = entities_to_displacy(entities, note.text)
    displacy.render(docs=displacy_data, manual=True, style="ent", options=options_displacy)


def get_note_diagnosis_pipeline(note_filepath, pipeline, visualize=False):
    note = TextDocument.from_file(Path(note_filepath))
    entities = pipeline.run([note.raw_segment])
    if visualize:
        visualize_entities(entities, note)
    if len(entities) ==0:
        return ""
    else:
        return entities
