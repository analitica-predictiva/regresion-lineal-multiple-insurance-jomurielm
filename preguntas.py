"""
Regresión Lineal Multiple
-----------------------------------------------------------------------------------------

En este laboratorio se entrenara un modelo de regresión lineal multiple que incluye la 
selección de las n variables más relevantes usando una prueba f.

"""
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import pandas as pd
import numpy as np


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
    # Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("insurance.csv", sep=",")

    # Asigne la columna `charges` a la variable `y`.
    y = df.charges.values

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy()

    # Remueva la columna `charges` del DataFrame `X`.
    X.pop('charges')

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12345. Use 300 patrones para la muestra de prueba.
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=(300/X.shape[0]),
        random_state=12345,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test


def pregunta_03():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe make_column_selector
    # Importe make_column_transformer
    # Importe SelectKBest
    # Importe f_regression
    # Importe LinearRegression
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe OneHotEncoder
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pipeline = Pipeline(
        steps=[
            # Paso 1: Construya un column_transformer que aplica OneHotEncoder a las
            # variables categóricas, y no aplica ninguna transformación al resto de
            # las variables.
            (
                "column_transfomer",
                make_column_transformer(
                    (
                        OneHotEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                    remainder="passthrough",
                ),
            ),
            # Paso 2: Construya un selector de características que seleccione las K
            # características más importantes. Utilice la función f_regression.
            (
                "selectKBest",
                SelectKBest(score_func=f_regression),
            ),
            # Paso 3: Construya un modelo de regresión lineal.
            (
                "LinearRegression",
                LinearRegression(),
            ),
        ],
    )

    # Cargua de las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Defina un diccionario de parámetros para el GridSearchCV. Se deben
    # considerar valores desde 1 hasta 11 regresores para el modelo
    param_grid = {
        "selectKBest__k": np.arange(1, 12, 1),
    }

    # Defina una instancia de GridSearchCV con el pipeline y el diccionario de
    # parámetros. Use cv = 5, y como métrica de evaluación el valor negativo del
    # error cuadrático medio.
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        refit=True,
        return_train_score=True,
    )

    # Búsque la mejor combinación de regresores
    gridSearchCV.fit(X_train, y_train)

    # Retorne el mejor modelo
    return gridSearchCV


def pregunta_04():
    """
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    # Importe mean_squared_error
    from sklearn.metrics import mean_squared_error

    # Obtenga el pipeline optimo de la pregunta 3.
    gridSearchCV = pregunta_03()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el modelo con los conjuntos de entrenamiento y prueba.
    y_train_pred = gridSearchCV.predict(X_train)
    y_test_pred = gridSearchCV.predict(X_test)

    # Compute el error cuadratico medio de entrenamiento y prueba. Redondee los
    # valores a dos decimales.

    mse_train = mean_squared_error(
        y_train,
        y_train_pred,
    ).round(2)

    mse_test = mean_squared_error(
        y_test,
        y_test_pred,
    ).round(2)

    # Retorne el error cuadrático medio para entrenamiento y prueba
    return mse_train, mse_test
