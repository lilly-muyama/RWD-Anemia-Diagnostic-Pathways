class FinalEntityFinder():
    def __init__(self):
        pass
        
    def run(self, entities):
        if len(entities) ==0:
            return []
        else:
            entities_not_to_use = []
            for entity in entities:
                section_attr = entity.attrs.get(label="section")[0].value
                family_attr = entity.attrs.get(label="family")[0].value
                negation_attr = entity.attrs.get(label="negation")[0].value
                hypothesis_attr = entity.attrs.get(label="hypothesis")[0].value
                if ((family_attr) | (negation_attr)) | (hypothesis_attr):
                    entities_not_to_use.append(entity)
            entities_to_use = [entity.text for entity in entities if entity not in entities_not_to_use]
            return entities_to_use 
        