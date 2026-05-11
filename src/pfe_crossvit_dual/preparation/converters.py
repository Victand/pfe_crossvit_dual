import pandas as pd
from pathlib import Path


def numbersToDf(numbersFilePath: str | Path = "data/full_data/Data_v2.numbers"):
    """
    Converts a .numbers file to a pandas.Dataframe object
    :param numbersFilePath: Description
    :type numbersFilePath: str | Path
    """
    from numbers_parser import Document

    doc = Document(numbersFilePath)
    sheets = doc.sheets
    tables = sheets[0].tables
    data = tables[0].rows(values_only=True)
    df = pd.DataFrame(data[1:], columns=data[0])

    print(
        f"Numbers file '{numbersFilePath}' successfully \nconverted to DataFrame. {len(df)} elements.\n"
    )
    return df
