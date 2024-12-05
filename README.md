
# Doctor Search LLM

This repository contains a prototype for a natural language search tool that helps administrative staff at Clalit Health Services clinics easily find relevant doctors and clinics from their service book data. The tool is developed in Python and uses a large language model (LLM) to enable searching using free-text queries in Hebrew.

## Features

- Load and preprocess doctor/clinic data from an Excel file
- Generate multilingual sentence embeddings for semantic search
- Save and load pre-computed embeddings for efficiency
- Interactive search interface for querying in natural language
- Integration with LLM for generating human-like responses based on search results

## Requirements

- Python 3.x
- pandas
- sentence-transformers
- scikit-learn
- llama-cpp

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/doctor-search-llm.git
   ```

2. Install the required dependencies:
   ```
   pip install pandas sentence-transformers scikit-learn llama-cpp
   ```

3. Place your `ServiceBook.xlsx` file in the project directory.

## Usage

1. Run the script:
   ```
   python medical-provider-search-ai.py
   ```

2. Enter your search query in Hebrew when prompted. For example:
   - "אני מחפש רופא אף-אוזן-גרון בחולון" (I'm looking for an ENT doctor in Holon)
   - "מה הכתובת והטלפון של פרופ' אבי שטיין?" (What is the address and phone number of Prof. Avi Stein?)

3. The tool will retrieve the top matching doctors/clinics and generate a natural language response using the LLM.

4. To exit the program, enter 'q' when prompted for a search query.

## How it works

1. The `ServiceBook.xlsx` file is loaded into a Pandas DataFrame, and the data is preprocessed and cleaned.

2. A multilingual sentence transformer model (`distiluse-base-multilingual-cased-v2`) is used to generate embeddings for the doctor/clinic information.

3. The generated embeddings are saved to a file (`doc_embeddings.pkl`) for future use. If the file already exists, the embeddings are loaded from the file instead of being regenerated.

4. The user enters a search query in Hebrew, and the query is converted into an embedding using the same sentence transformer model.

5. Cosine similarity is used to find the most similar doctor/clinic embeddings to the query embedding.

6. The top matching results are retrieved and formatted into a context string.

7. The context string and the user's query are passed to the LLM (`Qwen/Qwen2.5-3B-Instruct-GGUF`) to generate a natural language response.

8. The generated response is printed to the console, and the user is prompted for another search query.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
