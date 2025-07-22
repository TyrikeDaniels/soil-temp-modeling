from utils.model import evaluation_table, search, write_csv
from utils.preprocess import preprocess
import pandas as pd

def main():
    """
    Orchestrates the modeling process for varying seasons/depths.
    
    For each defined season (e.g., 'winter', 'summer'):
    - Loads and preprocesses the dataset specific to that season.
    - Performs nested cross-validation to find the best model per depth.
    - Evaluates and stores results in a summary table.
    
    The final evaluation table is saved as a CSV file.
    """
    
    # Define k-fold configurations
    k = 5

    # Initialize dictionary for logging
    seasonal_models = {}

    # Iterate through seasons
    for season in ["winter", "summer"]:

        # Filter data by season
        df = preprocess(season=season)

        # Perform GridSearchCV search for ensemble models on environmental data
        search_results = search(df=df, k=k, season=season)

        # Store search results into dictionary
        seasonal_models[season] = evaluation_table(search_results, season)

    # Combine season results into one dataframe and write into CSV file
    model_df = pd.concat(seasonal_models.values(), ignore_index=True)
    write_csv(model_df, "../results/model_summary_table.csv")

if __name__ == "__main__":
    main()