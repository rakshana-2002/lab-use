import pandas as pd

def load_dataset(file_path):
    """
    Load the dataset from a CSV file and return it as a pandas DataFrame.
    """
    try:
        dataset = pd.read_csv("D:\\23801928\\Gender_final.csv")
        return dataset
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        exit()

def predict_gender(name, dataset):
    """
    Predict the gender based on the dataset.
    """
    # Convert name to title case for matching
    name = name.title()
    
    # Search for the name in the dataset
    result = dataset[dataset['Name'] == name]
    
    # Check if a match is found
    if not result.empty:
        return result.iloc[0]['Gender']
    else:
        return "Unknown (Name not found in dataset)"

if __name__ == "__main__":
    # Load the dataset
    file_path = "names_dataset.csv"  # Update with your dataset path
    dataset = load_dataset("D:\\23801928\\Gender_final.csv")
    
    # Input name from the user
    name = input("Enter a name: ").strip()
    
    # Predict gender
    gender = predict_gender(name, dataset)
    print(f"The predicted gender for the name '{name}' is: {gender}")
