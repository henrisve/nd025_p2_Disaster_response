import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """A method used to load data into a dataframe

    Args:
        messages_filepath: Path to csv file that contaians messages 
        categories_filepath: Path to to csv file that contains categories
    Returns: 
        df: A pandas dataframe with both csv files combined
    """ 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Method that will clean up the dataFrame

    Args:
        df: input DataFrame

    Returns:
        df: clean Dataframe
    """ 
    categories = df['categories'].str.split(';',expand=True)
    categories.columns = categories.loc[0].str[:-2]
    categories = categories.apply(lambda x: (x.str[-1]).astype(int))
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df,categories],sort=False,axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df: pd.DataFrame, database_filename: str, table_name: str) -> None:
    """Save DataFrame to an SQLite database

    Args:
        df: dataFrame to be saved
        database_filename: Filename of database
        table_name: The name of the table
    """ 
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        table_name = "table1"
        save_data(df, database_filepath, table_name)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()