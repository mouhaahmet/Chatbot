import json
import os

def preprocess_data(input_dir, output_file):
    """Reads raw JSON files and processes them into a clean format."""
    conversations = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r') as file:
                data = json.load(file)
                # Assumes data has Q&A structure; adapt as needed
                for item in data:
                    question = item.get('question', '').strip()
                    answer = item.get('answer', '').strip()
                    if question and answer:
                        conversations.append({"question": question, "answer": answer})
    
    # Save processed data
    with open(output_file, 'w') as output:
        json.dump(conversations, output)
    print(f"Processed data saved to {output_file}")

