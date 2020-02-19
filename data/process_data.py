import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_file_path, categories_file_path):
    """
    this function is used to load the data from CSV files     
    
    Input:
    messages_file_path:   CSV file path for messages
    categories_file_path: CSV file path for categories
    Output:
    df: which contains merged  datasets  
    
    """
    messages  = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    
    df = pd.merge(categories, messages, on='id')
    
    return df



def clean_data(df):
    """
    this function is used to clean the dataset and split the categories column  into 36 features    
    Input:
    df: merged datasets  
    Output:
    df: merged datasets with new 36 categories columns
    
    """
    categories = df.categories.str.split(";",expand = True)
    row = categories.iloc[0].values
    clean_row = []
    for coloumn in row:
        first_clean =coloumn.replace("-0", "" )
        first_clean =first_clean.replace("-1", "")
        clean_row.append(first_clean)
    categories.columns  = clean_row
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-",expand = True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')   
    
    # drop the original column 
    df =df.drop(columns=['categories'])


    # concatenate the original dataframe
    df = pd.merge(df, categories, left_on="id", right_on=categories.index)

    # drop duplicates
    df= df.drop_duplicates()
    return df


def save_data(df, database_file_name):
    """
    this function is used to save the dataset into sqlite database
    Input:
    df: merged datasets  
    database_file_name: path to save the databse including the table name EX:  DisasterResponse.db

    Output:
    NONE     
    """
    engine = create_engine('sqlite:///{}'.format(database_file_name)) 
    db_file_name = database_file_name.split("/")[-1] 
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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