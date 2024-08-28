class Constants:
    def init(self, args):
        self.SEED = args.seed

        self.ACTION_SPACE = ["No anemia", "Vitamin B12/Folate deficiency anemia", "Unspecified anemia", "Anemia of chronic disease", "Iron deficiency anemia", "Other hemolytic anemia", 
        "Aplastic anemia", "Sickle cell anemia", "Inconclusive diagnosis", "hemoglobin", "gender", "mcv", "ferritin", "crp", "tsat", "reticulocytes", "haptoglobin", "b12", "folate", "sickle_cells"] 

        self.CLASS_DICT = {"No anemia":0, "Vitamin B12/Folate deficiency anemia":1, "Unspecified anemia":2, "Anemia of chronic disease":3, "Iron deficiency anemia":4, "Other hemolytic anemia":5, 
                           "Aplastic anemia":6, "Sickle cell anemia":7, "Inconclusive diagnosis":8}


        self.ACTION_NUM = len(self.ACTION_SPACE)

        self.CLASS_NUM = len(self.CLASS_DICT)

        self.FEATURE_NUM = self.ACTION_NUM - self.CLASS_NUM

        self.CHECKPOINT_FREQ = 1000000

        self.MAX_STEPS = self.FEATURE_NUM + 1

        self.BAD_DIAGNOSIS_REWARD = -1

        self.GOOD_DIAGNOSIS_REWARD = 1

        self.STEP_REWARD = -1/(2*self.FEATURE_NUM)

        self.MAX_STEP_REWARD = -1

        self.REPEATED_ACTION_REWARD = -1

constants = Constants()
