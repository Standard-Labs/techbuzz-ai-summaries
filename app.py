import streamlit as st
import pandas as pd
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import csv
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="AI Summary Generator", page_icon="📝", layout="wide")

###########################################
# CONSTANTS
###########################################

GPT4O_MODEL = "gpt-4o"
MAX_TOKENS = 1000
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default GPT-4o prompt template
DEFAULT_PROMPT_TEMPLATE = """
Shorten and summarize the following news article into one and a half lines including all the key information concisely in the following format:

[Article Title](URL) — The **Organization** did **something important** that has **significant impact**.

Examples:
[If It Looks Like a bank…](https://techcrunch.com/2024/11/21/apple-pay-paypal-cash-app-will-be-treated-more-like-banks/) — The **CFPB** just ruled that **any digital service** that handles a significant number of transactions (eg. **Apple**, **Paypal**) should be subject to **bank-like oversight**.

[**AI Boon Continues**](https://www.bloomberg.com/news/articles/2024-11-20/nvidia-forecast-fails-to-meet-the-loftiest-estimates-for-ai-star) — **Nvidia** beats **Q3** earnings but falls short of highest estimates. Forecasts show continued AI-driven growth of **70%** for Q4 as new chips begin to ship.

[**Monopoly Game Over**](https://www.politico.com/news/2024/11/20/doj-unveils-plan-to-breakup-googles-monopoly-00190753) — The **DOJ** seeks to force **Google** to sell its **Chrome** browser and **Android** mobile OS businesses to address illegal monopoly concerns.

IMPORTANT: Your response should be a single, complete, properly formatted summary following the examples above. Include the full article title in brackets with the URL as a link, followed by a concise summary with key entities in bold.

Description: {description}
URL: {url}
"""

###########################################
# DATA LOADING FUNCTIONS
###########################################

def load_tag_prompts():
    """
    Load tag-prompt mapping from tag_prompts.csv
    
    Returns:
        dict: A dictionary mapping tags to their corresponding prompts
    """
    tag_prompts = {}
    try:
        with open('tag_prompts.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                tag_prompts[row['Tag']] = row['Prompt']
    except Exception as e:
        st.error(f"Error loading tag_prompts.csv: {str(e)}")
    return tag_prompts

def find_column_names(df):
    """
    Find the Description, URL, and AI Summary Tag columns in the dataframe (case-insensitive)
    
    Args:
        df (pandas.DataFrame): The dataframe to search
        
    Returns:
        tuple: (description_col, url_col, tag_col) column names
    """
    description_col = None
    url_col = None
    tag_col = None
    
    for col in df.columns:
        if col.lower() == 'description':
            description_col = col
        elif col.lower() == 'url':
            url_col = col
        elif col.lower() == 'ai summary tag':
            tag_col = col
    
    return description_col, url_col, tag_col

###########################################
# SUMMARY GENERATION FUNCTIONS
###########################################

def get_prompt_for_tag(tag, tag_prompts):
    """
    Get the appropriate prompt template for a tag
    
    Args:
        tag (str): The tag to look up
        tag_prompts (dict): Dictionary mapping tags to prompts
        
    Returns:
        str: The prompt template to use
    """
    if tag and tag_prompts and tag in tag_prompts:
        return tag_prompts[tag]
    return DEFAULT_PROMPT_TEMPLATE

def generate_summary(description, url, api_key, prompt_template):
    """
    Generate a summary using OpenAI API with the given prompt template
    
    Args:
        description (str): The article description
        url (str): The article URL
        api_key (str): The OpenAI API key
        prompt_template (str): The prompt template to use
        
    Returns:
        str: The generated summary or error message
    """
    client = OpenAI(api_key=api_key)
    
    # Check if this is the default prompt (has placeholders) or a custom prompt from CSV
    if "{description}" in prompt_template and "{url}" in prompt_template:
        # Default prompt with placeholders - use format
        prompt = prompt_template.format(description=description, url=url)
    else:
        # Custom prompt from CSV - append the information
        prompt = f"{prompt_template}\n\nDescription: {description}\nURL: {url}"
    
    try:
        response = client.chat.completions.create(
            model=GPT4O_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles in a concise format. Always provide complete, properly formatted summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def validate_summary(summary):
    """
    Validate that a summary is complete and properly formatted
    
    Args:
        summary (str): The summary to validate
        
    Returns:
        bool: True if the summary is valid, False otherwise
    """
    # Check if summary contains a URL link
    if not ('[' in summary and '](' in summary and ')' in summary):
        return False
    
    # Check if summary contains the separator
    if ' — ' not in summary:
        return False
    
    # Check if summary contains bold text
    if '**' not in summary:
        return False
    
    return True

def process_single_row(row, description_col, url_col, tag_col, api_key, tag_prompts):
    """
    Process a single row and return the formatted summary
    
    Args:
        row (pandas.Series): The row to process
        description_col (str): The name of the description column
        url_col (str): The name of the URL column
        tag_col (str): The name of the tag column (or None)
        api_key (str): The OpenAI API key
        tag_prompts (dict): Dictionary mapping tags to prompts
        
    Returns:
        str: The formatted summary or error message
    """
    description = row[description_col]
    url = row[url_col]
    tag = row.get(tag_col) if tag_col else None
    
    prompt_template = get_prompt_for_tag(tag, tag_prompts)
    summary = generate_summary(description, url, api_key, prompt_template)
    
    if not validate_summary(summary):
        summary = "Invalid summary format. Please try again."
    
    return summary

def process_data(df, description_col, url_col, tag_col, api_key, tag_prompts, status_callback=None):
    """
    Process rows concurrently using ThreadPoolExecutor and return formatted summaries
    
    Args:
        df (pandas.DataFrame): The dataframe to process
        description_col (str): The name of the description column
        url_col (str): The name of the URL column
        tag_col (str): The name of the tag column (or None)
        api_key (str): The OpenAI API key
        tag_prompts (dict): Dictionary mapping tags to prompts
        status_callback (function): Callback function for progress updates
        
    Returns:
        list: List of formatted summaries
    """
    total_rows = len(df)
    completed = 0
    
    # Define a wrapper function to handle progress updates
    def process_with_progress(row_tuple):
        nonlocal completed
        i, row = row_tuple
        
        # Process the row
        summary = process_single_row(row, description_col, url_col, tag_col, api_key, tag_prompts)
        
        # Update progress (thread-safe way)
        if status_callback:
            completed += 1
            status_callback(completed-1, total_rows)
            
        return summary
    
    # Use ThreadPoolExecutor to process rows concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create an enumerated list of rows to keep track of indices
        row_tuples = list(df.iterrows())
        
        # Use executor.map to process rows concurrently
        results = executor.map(process_with_progress, row_tuples)
        
        # Collect results
        formatted_summaries = list(results)
    
    return formatted_summaries

###########################################
# UI RENDERING FUNCTIONS
###########################################

def render_header():
    """Render the app header"""
    st.title("AI Summary Generator")
    st.markdown("Upload a CSV with article descriptions and URLs to generate formatted summaries using GPT-4o.")

def render_input_section():
    """
    Render the input section (API key and file upload)
    
    Returns:
        tuple: (api_key, uploaded_file)
    """
    # Check if API key is available from environment
    if OPENAI_API_KEY:
        st.success("OpenAI API key loaded from environment variables.")
        api_key = st.text_input("Override OpenAI API key (optional):", type="password")
        # Use environment API key if no override provided
        if not api_key:
            api_key = OPENAI_API_KEY
    else:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    return api_key, uploaded_file

def render_preview(df):
    """
    Render a preview of the uploaded data
    
    Args:
        df (pandas.DataFrame): The dataframe to preview
    """
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

def render_results(formatted_summaries):
    """
    Render the formatted summaries
    
    Args:
        formatted_summaries (list): List of formatted summaries
    """
    st.success("Processing complete!")
    
    # Display formatted summaries in a markdown block
    st.markdown("### Formatted Summaries:")
    
    # Create a string with all formatted summaries
    all_summaries = "\n\n".join(formatted_summaries)
    
    # Display in a container
    formatted_container = st.container(border=True)
    with formatted_container:
        for summary in formatted_summaries:
            st.markdown(summary)
            st.markdown("\n\n")
    
    # Add a code block for easy copying
    st.code(all_summaries, language="markdown")
    st.info("👆 Copy the markdown above to use in your documents")

def render_instructions():
    """Render the instructions section"""
    st.markdown("---")
    st.markdown("""
    ### Instructions:
    1. Set the OPENAI_API_KEY environment variable or enter your API key in the field
    2. Upload a CSV file with 'Description' and 'URL' columns
    3. Optionally include an 'AI Summary Tag' column to use specific prompt templates
    4. Click 'Process with GPT-4o' to generate formatted summaries
    5. Copy the markdown text from the code block to use in your documents
    """)
    
    # Add sample format
    st.markdown("### Sample Format:")
    st.markdown("""
    The app will format each entry like:

    [Article Title](URL) — The **Organization** did **something important** that has **significant impact**.
    """)
    
    # Update sample CSV format information
    st.markdown("### Sample CSV Format:")
    st.code("""
    Article,URL,Description,Article Date,AI Summary Tag
    "Trump's Crypto Summit Sets Agenda for U.S. Pivot","https://www.coindesk.com/policy/2025/03/07/trump-s-crypto-summit-sets-agenda-for-u-s-pivot","President Trump hosted a crypto summit at the White House...","March 9, 2025","News"
    "Jobs report February 2025:","https://www.cnbc.com/2025/03/07/jobs-report-february-2025.html","Job growth in February 2025 was weaker than expected...","March 9, 2025","News"
    """)
    
    # Add information about available tags
    tag_prompts = load_tag_prompts()
    if tag_prompts:
        st.markdown("### Available AI Summary Tags:")
        tags_list = ", ".join([f"'{tag}'" for tag in tag_prompts.keys()])
        st.markdown(f"The following tags are available: {tags_list}")

def update_progress(current, total):
    """
    Update the progress bar and status text
    
    Args:
        current (int): Current progress (0-based index)
        total (int): Total number of items
    """
    progress = st.session_state.get('progress_bar')
    status = st.session_state.get('status_text')
    
    if progress is not None and status is not None:
        progress.progress((current + 1) / total)
        status.text(f"Processing row {current + 1}/{total}...")

###########################################
# MAIN APPLICATION
###########################################

def main():
    """Main application function"""
    # Render header
    render_header()
    
    # Render input section
    api_key, uploaded_file = render_input_section()
    
    # Initialize session state for progress tracking
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = None
    if 'status_text' not in st.session_state:
        st.session_state.status_text = None
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Render preview
            render_preview(df)
            
            # Load tag-prompt mapping
            tag_prompts = load_tag_prompts()
            
            # Find column names
            description_col, url_col, tag_col = find_column_names(df)
            
            if description_col is None or url_col is None:
                st.error("CSV must contain 'Description' and 'URL' columns (case insensitive).")
            else:
                # Process button
                process_button = st.button("Process with GPT-4o", key="process_button")
                
                if process_button:
                    if not api_key or api_key.strip() == "":
                        st.error("No OpenAI API key available. Please provide an API key or set the OPENAI_API_KEY environment variable.")
                    else:
                        # Create progress tracking elements
                        st.session_state.progress_bar = st.progress(0)
                        st.session_state.status_text = st.empty()
                        
                        # Process data
                        with st.spinner("Processing data with GPT-4o..."):
                            formatted_summaries = process_data(
                                df, 
                                description_col, 
                                url_col, 
                                tag_col,
                                api_key,
                                tag_prompts,
                                update_progress
                            )
                            
                            # Render results
                            render_results(formatted_summaries)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin.")
    
    # Render instructions
    render_instructions()

# Run the app
if __name__ == "__main__":
    main()
