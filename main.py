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


def generate_response(prompt, model_name):
    openai.api_key = openai.api_key
    messages = [
        {"role": "system", "content": "You are a professional excel file analyzer."},
        {"role": "user", "content": prompt},
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
                prompt = f"{query}\n{combined_data_text}"
                response = generate_response(prompt, self.model_name)

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
            summaries, graph_suggestions, questions_suggestions = perform_data_analysis_and_generate_questions(
                self.data_chunks, self.embeddings, self.model_name)
            self.output_text.delete(1.0, tk.END)
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