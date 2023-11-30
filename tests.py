"""
Calificacion del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import preguntas


def test_01():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 01
    pip3 install scikit-learn pandas numpy
    python3 tests.py 01
    """

    X, y = preguntas.pregunta_01()
    assert X.shape == (1338, 6)
    assert y.shape == (1338,)
    assert "charges" not in X.columns


def test_02():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 02
    pip3 install scikit-learn pandas numpy
    python3 tests.py 02
    (139, 10)
    -0.7869
    69.6029
    <class 'pandas.core.series.Series'>
    0.629
    """

    x_train, x_test, y_train, y_test = preguntas.pregunta_02()

    assert x_train.sex.value_counts().to_dict() == {"male": 536, "female": 502}
    assert x_test.sex.value_counts().to_dict() == {"female": 160, "male": 140}
    assert x_train.region.value_counts().to_dict() == {
        "southeast": 289,
        "northwest": 261,
        "southwest": 244,
        "northeast": 244,
    }

    assert x_test.region.value_counts().to_dict() == {
        "southwest": 81,
        "northeast": 80,
        "southeast": 75,
        "northwest": 64,
    }
    assert y_train.sum().round(2) == 13825369.07
    assert y_test.sum().round(2) == 3930455.92


def test_03():
    """
    ---< Run command >-----------------------------------------------------------------
    Pregunta 03
    pip3 install scikit-learn pandas numpy
    python3 tests.py 03
    """

    x_train, x_test, y_train, y_test = preguntas.pregunta_02()
    pipeline = preguntas.pregunta_03()

    assert pipeline.score(x_train, y_train).round(2) == -36943883.57
    assert pipeline.score(x_test, y_test).round(2) == -35336798.88


def test_04():
    """
    ---< Run command >--------------------------------------------------------------------
    Pregunta 04
    pip3 install scikit-learn pandas numpy
    python3 tests.py 04
    """

    mse_train, mse_test = preguntas.pregunta_04()

    assert mse_train == 36943883.57
    assert mse_test == 35336798.88


test = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
}[sys.argv[1]]

test()
