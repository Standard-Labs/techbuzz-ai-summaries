# AI Summary Generator

A Streamlit application that processes CSV files containing article descriptions and URLs, using OpenAI's GPT-4o to generate concise, formatted summaries.

## Features

- Upload CSV files with 'Description' and 'URL' columns
- Process each row through OpenAI's GPT-4o model
- Generate formatted summaries in a consistent style
- Download the processed CSV with an additional 'Formatted' column

## Requirements

- Python 3.7+
- OpenAI API key with access to GPT-4o

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Add your OpenAI API key to the `.env` file

```bash
cp .env.example .env
# Edit the .env file to add your OpenAI API key
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)
3. Upload a CSV file with 'Description' and 'URL' columns
4. Click "Process with GPT-4o" to generate formatted summaries
5. Copy the markdown text from the code block to use in your documents

Note: If you haven't set up the environment variable, you'll need to enter your OpenAI API key in the app.

## Sample CSV Format

Your input CSV should have at least these two columns:

```
Description,URL
"Apple Pay, PayPal, Cash App will be treated more like banks","https://techcrunch.com/2024/11/21/apple-pay-paypal-cash-app-will-be-treated-more-like-banks/"
"Nvidia forecast fails to meet the loftiest estimates for AI star","https://www.bloomberg.com/news/articles/2024-11-20/nvidia-forecast-fails-to-meet-the-loftiest-estimates-for-ai-star"
```

## Output Format

The app will add a 'Formatted' column with summaries in this format:

```
[Article Title](URL) — The **Organization** did **something important** that has **significant impact**.
```

## Environment Variables

The application uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for accessing the GPT-4o model

You can set these environment variables in a `.env` file in the root directory of the project, or you can set them directly in your environment.

## Note

This application requires an OpenAI API key with access to the GPT-4o model. API usage will incur charges according to OpenAI's pricing.
