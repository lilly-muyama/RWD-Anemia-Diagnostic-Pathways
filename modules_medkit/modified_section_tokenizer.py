from medkit.text.segmentation import SectionTokenizer

class ModifiedSectionTokenizer(SectionTokenizer):
    def __init__(self, section_dict, output_label):
        super().__init__(section_dict=section_dict, output_label=output_label)

    def run(self, segments):
        sections = super().run(segments)
        secs = [sec for sec in sections if sec.attrs.get(label="section")[0].value in ["motif", "conclusion"]]
        return secs