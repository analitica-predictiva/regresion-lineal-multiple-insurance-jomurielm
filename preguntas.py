"""
Regresión Lineal Multiple
-----------------------------------------------------------------------------------------

En este laboratorio se entrenara un modelo de regresión lineal multiple que incluye la 
selección de las n variables más relevantes usando una prueba f.

"""
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import pandas as pd


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
    # Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne la columna `charges` a la variable `y`.
    ____ = ____

    # Asigne una copia del dataframe `df` a la variable `X`.
    ____ = ____.____(____)

    # Remueva la columna `charges` del DataFrame `X`.
    ____.____(____)

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    # Importe train_test_split
    from ____ import ____

    # Cargue los datos y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12345. Use 300 patrones para la muestra de prueba.
    (X_train, X_test, y_train, y_test,) = ____(
        ____,
        ____,
        test_size=____,
        random_state=____,
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
    from ____ import ____

    pipeline = ____(
        steps=[
            # Paso 1: Construya un column_transformer que aplica OneHotEncoder a las
            # variables categóricas, y no aplica ninguna transformación al resto de
            # las variables.
            (
                "column_transfomer",
                ____(
                    (
                        ____(),
                        ____(____=____),
                    ),
                    remainder=____,
                ),
            ),
            # Paso 2: Construya un selector de características que seleccione las K
            # características más importantes. Utilice la función f_regression.
            (
                "selectKBest",
                ____(____=____),
            ),
            # Paso 3: Construya un modelo de regresión lineal.
            (
                "____",
                ____(____),
            ),
        ],
    )

    # Cargua de las variables.
    X_train, _, y_train, _ = pregunta_02()

    # Defina un diccionario de parámetros para el GridSearchCV. Se deben
    # considerar valores desde 1 hasta 11 regresores para el modelo
    param_grid = {
        ____: ____(____, ____),
    }

    # Defina una instancia de GridSearchCV con el pipeline y el diccionario de
    # parámetros. Use cv = 5, y como métrica de evaluación el valor negativo del
    # error cuadrático medio.
    gridSearchCV = ____(
        estimator=____,
        param_grid=____,
        cv=____,
        scoring=____,
        refit=____,
        return_train_score=____,
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
    from ____ import ____

    # Obtenga el pipeline optimo de la pregunta 3.
    gridSearchCV = pregunta_03()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el modelo con los conjuntos de entrenamiento y prueba.
    y_train_pred = ____.____(____)
    y_test_pred = ____.____(____)

    # Compute el error cuadratico medio de entrenamiento y prueba. Redondee los
    # valores a dos decimales.

    mse_train = ____(
        _____,
        _____,
    ).round(2)

    mse_test = ____(
        _____,
        _____,
    ).round(2)

    # Retorne el error cuadrático medio para entrenamiento y prueba
    return mse_train, mse_test
