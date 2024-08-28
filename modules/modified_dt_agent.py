import pandas as pd
from modules.env import AnemiaEnv
# from modules.constants import constants
from modules import former_constants as constants


class ModifiedDTAgent():
    def __init__(self, X, y):
        self.action_space = constants.ACTION_SPACE
        self.n_actions = constants.ACTION_NUM
        self.env = AnemiaEnv(X, y, random=False)
        self.hb_val = -1


    def get_action(self): # done
        if len(self.env.trajectory) == 0:
            next_action = 'hemoglobin'
        else:
            last_action = self.env.trajectory[-1]
            next_action = self.predict_next_action(last_action)
        try:
            next_action_index = self.env.actions.index(next_action)
        except:
            next_action_index = self.env.actions.index('Inconclusive diagnosis')
        return next_action_index

    def predict_next_action(self, last_action): # done
        last_action_idx = self.env.actions.index(last_action)
        last_action_val = self.get_feature_value(last_action_idx)
        if last_action == 'hemoglobin':
            self.hb_val = last_action_val
            action = self.predict_act_hb(last_action_val)
        elif last_action == 'gender':
            action = self.predict_act_gender(self.hb_val, last_action_val)
        elif last_action == 'mcv':
            action = self.predict_act_mcv(last_action_val)
        elif last_action == 'reticulocytes':
            action = self.predict_act_ret(last_action_val)
        elif last_action == 'ferritin':
            action = self.predict_act_ferritin(last_action_val)
        elif last_action == 'tsat':
            action = self.predict_act_tsat(last_action_val)
        elif last_action == 'crp':
            action = self.predict_act_crp(last_action_val)
        elif last_action== 'folate':
            b12_val = self.get_feature_value(self.env.actions.index('b12'))
            action = self.predict_act_folate(b12_val, last_action_val)
        elif last_action == 'b12':
            action = self.predict_act_b12(last_action_val)
        elif last_action == 'haptoglobin':
            action = self.predict_act_haptoglobin(last_action_val)
        elif last_action == 'sickle_cells':
            action = self.predict_act_sickle_cells(last_action_val)
        else:
            print('Invalid last action')
            raise Exception
        return action

    def get_feature_value(self, idx): # done
        if idx >= self.env.num_classes:
            feature_idx = idx-self.env.num_classes
            x = self.env.x.reshape(-1, constants.FEATURE_NUM)
            x_value = self.env.x[0, feature_idx]
            return x_value
        else:
            print('Last action cannot be a diagnosis action')

    def predict_act_hb(self, val): # done
        if val > 13:
            return 'No anemia'
        elif val< 0:
            print('Hemoglobin can\'t be negative')
            raise Exception
        # elif val <= 12:
        elif val < 12:
            return 'mcv'
        else:
            return 'gender'

    def predict_act_gender(self, hb_val, gender_val): # done
        if (hb_val < 0) | (gender_val< 0):
            print('Neither hemoglobin nor gender can be negative')
            raise Exception 
        # if (hb_val> 12) & (gender_val == 0):
        if (hb_val>= 12) & (gender_val == 0):
            return 'No anemia'
        else:
            return 'mcv'

    def predict_act_mcv(self, val): # done
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 80:
            return 'ferritin'
        elif val <= 100:
            return 'reticulocytes'
        elif val > 100:
            return 'b12'
        else:
            return 'Inconclusive diagnosis'

    def predict_act_ferritin(self, val): # done
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 30:
            return 'Iron deficiency anemia'
        elif val > 100:
            return 'Anemia of chronic disease'
        else:
            return 'crp'

    def predict_act_crp(self, val): # done
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 5:
            return 'Anemia of chronic disease'
        elif val >=5:
            return 'tsat'
        else:
            return 'Inconclusive diagnosis'

    def predict_act_tsat(self, val): # done
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val <= 20:
            return 'Iron deficiency anemia'
        elif val > 20:
            return 'Anemia of chronic disease' #try changing to unspecified anemia later
        else:
            return 'Inconclusive diagnosis'

    def predict_act_ret(self, val): # done
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val > 150:
            return 'haptoglobin'
        elif val <= 150:
            return 'Aplastic anemia'
        else:
            return 'Inconclusive diagnosis'

    
    def predict_act_haptoglobin(self, val): # done
        if val < 0:
            return 'Inconclusive diagnosis'
        elif val < 0.5:
            return 'sickle_cells'
            # return 'Hemolytic anemia'
        elif val >= 0.5:
            return 'Hemorragic anemia'
        else:
            return 'Inconclusive diagnosis'
    
    def predict_act_sickle_cells(self, val):
        if val <= 0:
            return 'Other hemolytic anemia'
        elif val == 1:
            return 'Sickle cell anemia'
        else:
            return 'Inconclusive diagnosis'


    def predict_act_b12(self, val): # done
        if val < 200:
            return 'Vitamin B12/Folate deficiency anemia'
        else: 
            return 'folate'

    def predict_act_folate(self, b12_val, folate_val): # done
        if folate_val < 2:
            return 'Vitamin B12/Folate deficiency anemia'
        elif (folate_val >= 2) | (b12_val > 200):
            return 'Unspecified anemia'
        # elif (b12_val < 0) & (folate_val < 0):
        #     return 'Inconclusive diagnosis'
        else:
            return 'Inconclusive diagnosis'      


    def test(self):
        test_df = pd.DataFrame()
        try:
            while True:
                obs, done = self.env.reset(), False
                while not done:
                    action = self.get_action()
                    obs, rew, done, info = self.env.step(action)
                    if done == True:
                        test_df = test_df.append(info, ignore_index=True)
        except StopIteration:
            print('Testing done.....')
        return test_df

    def test_sample(self, idx):
        try:
            obs, done = self.env.reset(idx=idx), False
            while not done:
                action = self.get_action()
                obs, rew, done, info = self.env.step(action)
                if done==True:
                    return info['trajectory']
        except Exception as e:
            print(e)
