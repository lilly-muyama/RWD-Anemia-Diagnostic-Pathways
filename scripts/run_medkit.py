import pandas as pd
import glob, os, sys
import argparse
sys.path.append("..")
from modules_medkit.constants import constants
from medkit.text.context import FamilyDetector, NegationDetector, HypothesisDetector

parser = argparse.ArgumentParser(description = "Parameters for dqn model")
parser.add_argument("-t", "--note_type", help="Type of note - either 'crh' or 'crc'", type=str, default="crh")
args = parser.parse_args()
constants.init(args)

from modules_medkit import utils

note_type = constants.NOTES_TYPE.upper()
folder = f"../data/{note_type}"
terms_path = constants.TERMS_FILEPATH
terms_col = constants.TERMS_COLUMN


if args.note_type.lower() == 'crc':
        section_dict = constants.CRC_SECTION_DICT
elif args.note_type.lower() == 'crh':
    section_dict = constants.CRH_SECTION_DICT
else:
    print('Unknown note type')
    sys.exit()

pipeline = utils.define_pipeline(section_dict=section_dict)


if __name__ == "__main__":
    print('STARTING....')
    df = pd.DataFrame(columns=["filename", "diagnosis"])
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            filepath = os.path.join(folder, file)
            print(filepath)
            entities = utils.get_note_diagnosis_pipeline(filepath, pipeline)
            new_row = pd.DataFrame({"filename": [file], "diagnosis": [entities]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(f"../results/{note_type}_results.csv", index=False)
    print("All done. Results have been saved in the results folder.")
