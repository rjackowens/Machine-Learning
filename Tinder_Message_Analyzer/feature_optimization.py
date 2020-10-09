import pandas as pd
import boto3
import json


def feature_optimization(csv_file, export_name="export_dataframe.csv"):

    dataset = pd.read_csv(csv_file)

    # populating number_of_characters
    for index, row in dataset.iterrows():
        stripped_str = row["message"].replace(" ", "")
        num_chars = len(stripped_str)
        dataset.at[index, ["number_of_characters"]]=num_chars

    session = boto3.Session()
    comprehend = session.client("comprehend")


    def get_sentiment(text_input: str):
        sentiment_results = comprehend.detect_sentiment(Text=text_input, LanguageCode="en")
        test = sentiment_results["SentimentScore"]["Positive"]
        return str(test)[:4]


    # populating sentiment
    for index, row in dataset.iterrows():
        sentiment = get_sentiment(row["message"])
        dataset.at[index, ["sentiment"]]=sentiment


    dataset.to_csv(export_name, index = False, header=True)


if __name__ == "__main__":
    print("Running feature optimization...")
    feature_optimization("opening_messages.csv")
