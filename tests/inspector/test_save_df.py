from io import BytesIO, StringIO

import pandas as pd

from giskard.io_utils import compress, decompress, save_df


def test_save_df_titanic():
    df = pd.read_csv(
        StringIO(
            """
        PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
        2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
        3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
        """
        )
    )
    csv_compressed = compress(save_df(df))
    df_reloaded = pd.read_csv(BytesIO(decompress(csv_compressed)))
    pd.testing.assert_frame_equal(df, df_reloaded)
