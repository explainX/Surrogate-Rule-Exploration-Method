import os
import sys
from pathlib import Path
from sys import platform
import subprocess

path = Path(__file__).parent.absolute()
path_dataset = os.path.join(path, "datasets")
path = os.path.join(path, "lib")
sys.path.append(path)

from imports import *


"""
This class calculates feature importance

Input: 


"""


class explainx_pro():
    def __init__(self):
        super(explainx_pro, self).__init__()
        self.param = {}

    # is classification function?


    def rule_exploration(self, df, y, model):
        from apps.webapp.server.server import run
        y_pred = model.predict(df)
        target_names = list(set(y_pred))
        target_names = list(map(int, target_names))
        X = df.drop(columns=['y'], errors='ignore').values
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        self.rule_output_data(cols=df.drop(columns=['y'], errors='ignore').columns.values.tolist(),
                              data=X.tolist(),
                              target_names=target_names,
                              real_min=min_val.tolist(),
                              real_max=max_val.tolist(),
                              y_pred=y_pred.tolist(),
                              y_gt=y)
        # print(os.system('python apps.webapp.server.py'))
        run()

    def get_random_string(self, length):
        letters = string.ascii_lowercase + string.ascii_uppercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def rule_output_data(self, cols, data, target_names, real_min, real_max, y_pred, y_gt):
        # data_name = self.get_random_string(5)
        filename = "{}/apps/prepare/output/".format(path) + "user_defined" + "/test.json"
        filename2 = "{}/apps/webapp/data/".format(path) + "user_defined" + "/test.json"

        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        to_output = {}
        to_output['columns'] = cols
        to_output['data'] = data
        to_output['target_names'] = target_names
        to_output['real_min'] = real_min
        to_output['real_max'] = real_max
        to_output['y_pred'] = y_pred
        to_output['y_gt'] = y_gt
        with open(filename, 'w') as output:
            output.write(json.dumps(to_output))
        with open(filename2, 'w') as output:
            output.write(json.dumps(to_output))

    def dataset_boston(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.boston()
        return X, y

    def dataset_iris(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.iris()
        return X, y

    def dataset_heloc(self):
        dataset = pd.read_csv(path_dataset + "/heloc_dataset.csv")

        map_riskperformance = {"RiskPerformance": {"Good": 1, "Bad": 0}}
        dataset.replace(map_riskperformance, inplace=True)
        y = list(dataset["RiskPerformance"])
        X = dataset.drop("RiskPerformance", axis=1)
        return X, y

explainx_pro = explainx_pro()


