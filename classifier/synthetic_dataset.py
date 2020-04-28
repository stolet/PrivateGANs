import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

class SyntheticCelebA():

    def __init__(self, main_folder='../synthetic_data/'):
        self.main_folder = main_folder
        self.old_folder = os.path.join(main_folder, 'celeba_old/')
        self.young_folder = os.path.join(main_folder, 'celeba_young/')
        self._parse_data()
    
    def _parse_data(self):
        old_ids = ['celeba_old/' + s for s in os.listdir(self.old_folder)]
        young_ids = ['celeba_young/' + s for s in os.listdir(self.young_folder)]
        
        old_data = {"image_id": old_ids, "Young": np.zeros(len(old_ids))}
        young_data = {"image_id": young_ids, "Young": np.ones(len(young_ids))}
        
        df_o = pd.DataFrame(old_data, columns = ["image_id", "Young"])
        df_o.set_index("image_id", drop=False, inplace=True)

        df_y = pd.DataFrame(young_data, columns = ["image_id", "Young"])
        df_y.set_index("image_id", drop=False, inplace=True)
        df = pd.concat([df_o, df_y])
        df = df.sample(frac=1, axis=0)
        self.dataset = df

