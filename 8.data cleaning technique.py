import pandas as pd
def display_head(df):
    print("\n Head of the DataFrame:")
    print(df.head(), "\n")

def display_tail(df):
    print("\n Tail of the DataFrame:")
    print(df.tail(), "\n")

def drop_na(df):
    df_dropped = df.dropna()
    print("\n DataFrame after dropping rows with missing values:")
    print(df_dropped, "\n")

def fill_na(df):
    mean_value_Salary = df['Salary'].mean() 
    mean_value_Age = df['Age'].mean() 
    df['Salary'].fillna(value=mean_value_Salary, inplace=True) 
    df['Age'].fillna(value=mean_value_Age, inplace=True) 
    print('Updated Dataframe:') 
    print("\n DataFrame after filling missing values:")
    print(df['Age'],df['Salary'], "\n")
    
def check_isnull(df):
    print("\n Checking for null values:")
    print(df.isnull().sum(), "\n")

def main():
    print("\n\t\t Exploring Data Cleaning Techniques With Python ")
    print("\t\t ---------------------------------------------- ")
    df = pd.read_csv('data.csv')
    while True:
        print("\n\t\t\t\t MENU")
        print("\t\t\t\t ----")
        print("\t\t\t1. Display Head")
        print("\t\t\t2. Display Tail")
        print("\t\t\t3. Drop NA")
        print("\t\t\t4. Fill NA")
        print("\t\t\t5. Check for Null Values")
        choice = input("Enter your choice: ")
        if choice == '1':
             display_head(df)
  
        elif choice == '2':
            display_tail(df)

        elif choice == '3':
            drop_na(df)

        elif choice == '4':
            fill_na(df)

        elif choice == '5':
            check_isnull(df)
            break
        else:
            print("Invalid choice. Please try again.\n")
            continue
        ch = input("Do you want to continue? (1/0): ")
        if ch != '1':
            print("Exiting the program")
            break
        

if __name__ == "__main__":
    main()
