from models.train import train_model

def main():
    data_file = 'data/training_data.csv'
    input_length = 20  # Assuming a fixed input length for simplicity
    
    # Train the chatbot model
    train_model(data_file, input_length)

    print("Training complete. Model is ready to use.")

if __name__ == "__main__":
    main()
