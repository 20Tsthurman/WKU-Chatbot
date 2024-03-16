from models.train import train_model

def main():
    try:
        print("Starting training...")
        data_file = 'data/intents.json'  # Ensure this path is correct
        
        # Verify the data file exists before proceeding
        import os
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print("Please check the file path and try again.")
            return  # Exit the function if the data file is not found
        
        # Train the chatbot model
        train_model(data_file)
        print("Training complete. Model is ready to use.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()
