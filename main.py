import pandas as pd
import matplotlib.pyplot as plt
import openai
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import textwrap
from concurrent.futures import ThreadPoolExecutor
import tensorflow_hub as hub
import numpy as np
from sklearn.neighbors import NearestNeighbors
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tkinter import ttk, filedialog, simpledialog
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

def set_api_key(self):
    api_key = simpledialog.askstring("API Key", "Please enter your OpenAI API key:")
    if api_key:
        openai.api_key = api_key
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "API key has been set.")
    else:
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Invalid API key.")

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def read_excel_file(file_path):
    data = pd.read_excel(file_path)
    return data

def split_data(data, chunk_size):
    num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)
    data_chunks = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    return data_chunks

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def get_embeddings(data_chunks):
    embeddings = []
    for chunk in data_chunks:
        chunk_text = preprocess_text(chunk.to_string())
        chunk_embedding = embed([chunk_text])[0].numpy()
        embeddings.append(chunk_embedding)
    return embeddings

def get_most_relevant_chunks(query, embeddings, k=5):
    query_embedding = embed([query])[0].numpy()
    nn = NearestNeighbors(n_neighbors=min(k, len(embeddings)))
    nn.fit(embeddings)
    indices = nn.kneighbors([query_embedding], return_distance=False)[0]
    return indices


def get_data_info(data):
    data_info = []
    for column in data.columns:
        column_info = {"name": column, "type": data[column].dtype}

        # Calculate basic statistics for numeric columns
        if np.issubdtype(data[column].dtype, np.number):
            column_info["mean"] = data[column].mean()
            column_info["std"] = data[column].std()
            column_info["min"] = data[column].min()
            column_info["25%"] = data[column].quantile(0.25)
            column_info["50%"] = data[column].quantile(0.50)
            column_info["75%"] = data[column].quantile(0.75)
            column_info["max"] = data[column].max()

        # Calculate frequency for categorical columns
        else:
            column_info["value_counts"] = data[column].value_counts().to_dict()

        data_info.append(column_info)

    return data_info


def generate_response(prompt, model_name, data_info):
    openai.api_key = openai.api_key
    data_info_str = "\n".join([f"Column: {info['name']}, Data Type: {info['type']}" for info in data_info])
    messages = [
        {"role": "system", "content": "You are a professional excel file analyzer."},
        {"role": "user", "content": f"{prompt}\n\nData Information:\n{data_info_str}"},
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=512,
        n=1,
        temperature=0.8,
    )
    message = response['choices'][0]['message']['content'].strip()
    return message


def generate_questions_suggestions(prompt, model_name):
    openai.api_key = openai.api_key
    messages = [
        {"role": "system", "content": "You are an AI assistant that suggests questions to ask about a dataset."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=100,
        n=1,
        temperature=0.8,
    )
    message = response['choices'][0]['message']['content'].strip()
    return message

def perform_data_analysis_and_generate_questions(data_chunks, embeddings, model_name):
    summaries, graph_suggestions = analyze_data(data_chunks, embeddings)
    prompt = "Based on the data summary, suggest some questions to ask:"
    questions_suggestions = generate_questions_suggestions(prompt, model_name)
    return summaries, graph_suggestions, questions_suggestions

def analyze_data(data_chunks, embeddings):
    # Get the most relevant chunks based on the user query
    query = "Analyze the following data and provide a summary in verbally and graph suggestions after you have read all the chunks. This data is part of a larger dataset:"
    relevant_chunk_indices = get_most_relevant_chunks(query, embeddings)
    relevant_chunks = [data_chunks[i] for i in relevant_chunk_indices]

    # Combine all relevant chunks into a single string
    combined_data_text = "\n".join(chunk.to_string() for chunk in relevant_chunks)

    # Split the combined_data_text into smaller pieces to stay within the token limit
    data_text_parts = textwrap.wrap(combined_data_text, width=2048, break_long_words=False)

    summaries = []
    graph_suggestions = set()

    def analyze_chunk(data_text_part):
        messages = [
            {"role": "system", "content": "You are a professional excel file analyzer"},
            {"role": "user", "content": f"{query}\n{data_text_part}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=250,
            n=1,
            temperature=0.8,
        )
        response_content = response['choices'][0]['message']['content'].strip()
        return response_content

    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(analyze_chunk, data_text_parts))

    for response_content in responses:
        if "\nGraph Suggestions:" in response_content:
            summary, graph_suggestion = response_content.split("\nGraph Suggestions:")
            summaries.append(summary.strip())
            graph_suggestions.add(graph_suggestion.strip())
        else:
            summaries.append(response_content)
            graph_suggestions.add("No graph suggestions provided.")

            return "\n".join(summaries), "\n".join(graph_suggestions)

        def create_graphs(data, graph_suggestions):
            suggestions = graph_suggestions.split("\n")

            for suggestion in suggestions:
                if " using " not in suggestion:
                    continue

                # Extract the column names and graph type from the suggestion
                graph_type, columns = suggestion.split(" using ")[0].strip(), suggestion.split(" using ")[
                    1].strip().split(" and ")
                x_column, y_column = columns

                if graph_type.lower() == "bar graph":
                    plt.bar(data[x_column], data[y_column])
                elif graph_type.lower() == "scatter plot":
                    plt.scatter(data[x_column], data[y_column])
                elif graph_type.lower() == "line plot":
                    plt.plot(data[x_column], data[y_column])

                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f"{graph_type} - {x_column} vs {y_column}")
                plt.show()

def create_visualizations(graph_suggestions, data, limit=3):
    counter = 0
    for suggestion in graph_suggestions:
        # Extract the graph type and columns involved
        graph_type = suggestion["type"]
        columns = suggestion["columns"]

        if graph_type == "scatter":
            plt.figure()
            sns.scatterplot(data=data, x=columns[0], y=columns[1])
            plt.title(f"{columns[0]} vs {columns[1]}")
            plt.show()
            counter += 1

        # Add other graph types and their plotting code here, e.g. line plot, bar plot, etc.

        if limit is not None and counter >= limit:
            break


# Add the parse_graph_suggestions function
def parse_graph_suggestions(graph_suggestions_text):
    graph_suggestions_list = []
    for line in graph_suggestions_text.split("\n"):
        if not line.strip():
            continue
        try:
            graph_type, columns = line.split(" using ")
            graph_suggestions_list.append({"type": graph_type.strip().lower(), "columns": columns.strip().split(" and ")})
        except ValueError:
            print(f"Warning: Unable to parse graph suggestion line: '{line}'")
            continue
    return graph_suggestions_list


def summarize_data(data_chunk):
    data_info = get_data_info(data_chunk)
    description = "Data summary:\n"
    for column in data_info:
        column_name = column['name']
        column_type = column['type']
        description += f"Column: {column_name}, Type: {column_type}\n"

        # Add basic statistics for numeric columns
        if np.issubdtype(column['type'], np.number):
            description += f"Mean: {column['mean']:.2f}, Std: {column['std']:.2f}, Min: {column['min']}, 25%: {column['25%']}, 50%: {column['50%']}, 75%: {column['75%']}, Max: {column['max']}\n"

        # Add frequency for categorical columns
        else:
            value_counts = ", ".join(f"{key}: {value}" for key, value in column['value_counts'].items())
            description += f"Value counts: {value_counts}\n"

        description += "\n"

    return description


def generate_basic_graph_suggestions(data, max_graphs=3):
    numeric_columns = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]

    graph_suggestions = []
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            graph_suggestions.append({"type": "scatter", "columns": [numeric_columns[i], numeric_columns[j]]})
            if len(graph_suggestions) >= max_graphs:
                break
        if len(graph_suggestions) >= max_graphs:
            break

    return graph_suggestions


def perform_time_series_analysis(data):
    # Identify the date column
    date_column = None
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.datetime64):
            date_column = column
            break

    if not date_column:
        print("No date column found.")
        return None

    # Set the date column as the index and sort by date
    data = data.set_index(date_column).sort_index()

    # Resample the data by calculating the mean for each month
    monthly_data = data.resample('M').mean()

    # Print the resampled data
    print("Monthly data summary:\n", monthly_data)

    # Calculate basic statistics for numeric columns
    numeric_columns = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]
    summary_stats = data[numeric_columns].describe()

    return summary_stats

def analyze_categorical_data(data):
    # Identify categorical columns
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

    # Calculate the frequency distribution for each categorical column
    categorical_data_summary = {}
    for column in categorical_columns:
        categorical_data_summary[column] = data[column].value_counts().to_dict()

    return categorical_data_summary



class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("GPT-4 Excel Analyzer")
        self.geometry("600x400")

        self.data = None
        self.data_chunks = None
        self.embeddings = None
        self.qa_window = None

        main_frame = ttk.Frame(self)
        main_frame.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)
        ttk.Button(button_frame, text="Set API Key", command=self.set_api_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Set Model", command=self.set_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Open File", command=open_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Data Analysis", command=self.perform_data_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Ask Question", command=self.ask_question).pack(side=tk.LEFT, padx=5)
        self.model_name = "gpt-3.5-turbo"  # Add this line to set the default model

        self.model_combobox = ttk.Combobox(button_frame, values=["gpt-3.5-turbo", "gpt-4"])
        self.model_combobox.set(self.model_name)
        self.model_combobox.bind("<<ComboboxSelected>>", self.model_selection)
        self.model_combobox.pack(side=tk.RIGHT, padx=5)  # Adjust the side and padx values as needed

        global status_label
        status_label = ttk.Label(main_frame, text="No file loaded")
        status_label.pack(padx=5, pady=5)

        self.output_text = tk.Text(main_frame, wrap=tk.WORD)
        self.output_text.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)

    def model_selection(self, event=None):
        self.model_name = self.model_combobox.get()

    def set_api_key(self):
        api_key = simpledialog.askstring("API Key", "Please enter your OpenAI API key:")
        if api_key:
            openai.api_key = api_key
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "API key has been set.")
        else:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Invalid API key.")

    def set_model(self):
        model = simpledialog.askstring("Model", "Please enter the model name (e.g., gpt-3.5-turbo, gpt-4):")
        if model:
            self.model_name = model
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Model has been set to {model}.")
        else:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Invalid model name.")

    def ask_question(self):
        if self.embeddings and hasattr(self, 'model_name'):
            query = simpledialog.askstring("Question", "Please enter your question:")
            if query:
                relevant_chunk_indices = get_most_relevant_chunks(query, self.embeddings)
                relevant_chunks = [self.data_chunks[i] for i in relevant_chunk_indices]

                combined_data_text = "\n".join(chunk.to_string() for chunk in relevant_chunks)

                # Split the query into smaller chunks based on sentences using NLTK
                chunk_prompts = nltk.sent_tokenize(query)
                data_info = get_data_info(self.data)

                responses = []
                for chunk_prompt in chunk_prompts:
                    # Generate response for each chunk prompt
                    chunk_response = generate_response(chunk_prompt, self.model_name, data_info)
                    responses.append(chunk_response)

                response = " ".join(responses)

                self.show_question_answer_window(query, response)
        else:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Please set API key, model, and load an Excel file.")

    def show_question_answer_window(self, question, answer):
        if self.qa_window is None:  # Create a new Toplevel window if it doesn't exist
            self.qa_window = tk.Toplevel(self)
            self.qa_window.title("Question and Answer")
            self.qa_window.geometry("600x400")

            self.qa_output_text = tk.Text(self.qa_window, wrap=tk.WORD)  # Store the Text widget as an instance variable
            self.qa_output_text.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)

        #self.qa_output_text.delete(1.0, tk.END)  # Clear the existing content
        self.qa_output_text.insert(tk.END, f"Question: {question}\n")
        self.qa_output_text.insert(tk.END, f"Answer from GPT-4:\n{answer}")

    def perform_data_analysis(self):
        if self.data_chunks and self.embeddings:
            # Summarize the entire dataset
            dataset_summary = summarize_data(self.data)

            summaries, graph_suggestions, questions_suggestions = perform_data_analysis_and_generate_questions(
                self.data_chunks, self.embeddings, self.model_name)

            # Parse the graph suggestions from the AI response
            graph_suggestions_list = parse_graph_suggestions(graph_suggestions)
            if not graph_suggestions_list:
                graph_suggestions_list = generate_basic_graph_suggestions(self.data, max_graphs=3)

            # Call the create_visualizations function with the parsed suggestions and data
            create_visualizations(graph_suggestions_list, self.data)
            # Call the perform_time_series_analysis function
            time_series_summary = perform_time_series_analysis(self.data)
            if time_series_summary is not None:
                self.output_text.insert(tk.END, "\nTime Series Summary:\n")
                self.output_text.insert(tk.END, time_series_summary.to_string())

            # Call the analyze_categorical_data function
            categorical_data_summary = analyze_categorical_data(self.data)
            self.output_text.insert(tk.END, "\nCategorical Data Summary:\n")
            for column, value_counts in categorical_data_summary.items():
                value_counts_str = ", ".join(f"{key}: {value}" for key, value in value_counts.items())
                self.output_text.insert(tk.END, f"{column}: {value_counts_str}\n")
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, dataset_summary)
            self.output_text.insert(tk.END, "Summary from GPT-4:\n")
            self.output_text.insert(tk.END, summaries)
            self.output_text.insert(tk.END, "\n\nGraph Suggestions from GPT-4:\n")
            self.output_text.insert(tk.END, graph_suggestions)
            self.output_text.insert(tk.END, "\n\nQuestion Suggestions from GPT-4:\n")
            self.output_text.insert(tk.END, questions_suggestions)
        else:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Please load an Excel file first.")


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if file_path:
        data = read_excel_file(file_path)
        app.data = data
        app.data_chunks = split_data(data, chunk_size=50)
        app.embeddings = get_embeddings(app.data_chunks)
        status_label.config(text=f"Loaded file: {file_path}")

def main():
    global app
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()