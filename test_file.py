from hx_deep_learning_tools import HxDenseNeuralNetworkBinaryClassifier
from constants_and_tools import ConstantsAndTools
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd


class Test:
    def __init__(self):
        self.CT: ConstantsAndTools = ConstantsAndTools()

        self.dnn_bounds: dict = {
            'units': 248,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 56,
            'regularization': 0.001,
            'hidden_layers': 1,  # {0: <, 1: >, 2: <>, 3: ><}
            'layers_number': 2  # En caso de excesivos recursos tirará error
        }

        self.xgb_bounds: dict = {
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 300,
            'reg_alpha': 1,
            'reg_lambda': 1,
        }

        self.cat_bounds: dict = {
            'iterations': 250,
            'max_depth': 3,
            'learning_rate': 0.005,
            'l2_leaf_reg': 1,
            'penalties_coefficient': 1
        }

    def train_classifiers(self):
        # -- 0: DATA--------------------------------

        # Cargar dataset (ya viene limpio y preparado para clasificación binaria)
        dataset = load_breast_cancer()

        # Pasar a DataFrame para usar como tus otros datasets
        x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.DataFrame(dataset.target, columns=["target"])

        # Dividir en train/test si lo necesitas
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Crear diccionario como espera tu clase
        data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

        # -- 1: LGBM
        dnn = HxDenseNeuralNetworkBinaryClassifier(data_dict,
                                                   self.dnn_bounds,
                                                   None,
                                                   'accuracy',
                                                   'binary')
        dnn.fit_and_get_model_and_results()
        # lgbm.execute_shap_analysis()



    def train_regressors(self):

        # -- 0: DATA    ------------------------------------------------

        # Cargar dataset de diabetes (ya normalizado y sin missing values)
        dataset = load_diabetes()

        # Convertir a DataFrame para seguir el mismo formato que usas
        x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.DataFrame(dataset.target, columns=["target"])

        # Dividir en train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Crear diccionario en el formato esperado
        data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

        bins: dict = {
            "bins": [0, 50, 100, 150, 200, 250, 300, 350, 400],
            "labels": ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300-350", "350-400"]
        }

        # -- 1: LGBM
        """lgbm = HxLightGbmRegressor(data_dict, self.lgbm_bounds, None, 'mae','regression', bins=bins)
        lgbm.fit_and_get_model_and_results()
        lgbm.execute_shap_analysis()

        # -- 2: XGB
        xgb = HxXtremeGradientBoostingRegressor(data_dict, self.xgb_bounds, None, 'mse','reg:squarederror', bins=bins)
        xgb.fit_and_get_model_and_results()
        xgb.execute_shap_analysis()

        # -- 3: CAT
        cat = HxCatBoostRegressor(data_dict, self.cat_bounds, None, 'mse','RMSE', bins=bins)
        cat.fit_and_get_model_and_results()
        cat.execute_shap_analysis()"""


Test().train_classifiers()