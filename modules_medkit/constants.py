class Constants:
    def init(self, args):
        self.NOTES_TYPE = args.note_type
        self.CRH_SECTION_DICT = {
        "introduction": ["Service de médecine interne"],
        "destinataires": ["DESTINATAIRES"],
        "motif": ["MOTIF D’HOSPITALISATION"],
        "mode de vie":["MODE DE VIE"],
        "antécédents":["ANTÉCÉDENTS", "ANTECEDENTS", "ANTÉCEDENTS", "ANTECÉDENTS"],
        "traitement_entree":["TRAITEMENT A L’ENTREE", "TRAITEMENT A L’ENTRÉE"],
        "histoire":["HISTOIRE DE LA MALADIE"],
        "examens_entree":["EXAMEN CLINIQUE À L’ENTRÉE", "EXAMEN CLINIQUE À L’ENTREE"],
        "examens_complementaires": ["EXAMENS COMPLÉMENTAIRES", "EXAMENS COMPLEMENTAIRES"],
        "imagerie": ["IMAGERIE"],
        "evolution": ["ÉVOLUTION DANS LE SERVICE", "EVOLUTION DANS LE SERVICE"],
        "conclusion":["CONCLUSION", "Conclusion", "conclusion"],
        "suivi": ["Suivi prévu"],
        "traitement_sortie": ["TRAITEMENT DE SORTIE"],
        "destination": ["DESTINATION"],
        "prochaines_consultations": ["PROCHAINES CONSULTATIONS"],
        "transfusion": ["TRANSFUSION SANGUINE"],
        "protocole": ["INCLUSION DANS UN PROTOCOLE DE RECHERCHE CLINIQUE"]
        }

        self.CRC_SECTION_DICT = {
            "introduction": ["COMPTE RENDU DE CONSULTATION"],
            "motif": ["Motif de consultation"],
            "antécédents":["Antécédents", "Antecedents", "Antecédents", "Antécedents"],
            "antécédents_familiaux":["Antécédents familiaux"],
            "allergies": ["Allergies"],
            "traitement_actuel":["Traitement actuel"],
            "mode de vie":["Mode de vie"],        
            "histoire":["Histoire de la maladie"],
            "conclusions_initiales": ["Conclusions"],
            "examens":["Examen physique"],
            "examens_complementaires": ["Examens complémentaires", "Examens complémentaires"],
            "imagerie": ["Imagerie"],
            "epreuves_fonctionelles": ["Épreuves fonctionnelles"],
            "biologie": ["Biologie ce jour"],
            "biochemie": ["BIOCHIMIE GENERALE SANG"],
            "hematologie": ["HEMATOLOGIE/CYTOLOGIE"],
            "conclusion":["Conclusion", "CONCLUSION", "conclusion"],
            "suivi": ["Suivi prévu"],
            "traitement_prescrit": ["Traitement prescrit"],
            "fin": ["Bien confraternellement,"],
            "adresse": ["rue Leblanc"]
        }

        
        self.TERMS_FILEPATH = "../data/anemias.csv"
        self.TERMS_COLUMN = "anemia synonym"

constants = Constants()
