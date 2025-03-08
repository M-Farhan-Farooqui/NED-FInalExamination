import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def research_and_write(topic, urls):
    """
    Conducts research on a given topic by scraping information from specified URLs,
    synthesizes the information using a generative AI model, and produces a
    well-structured research report.
    """

    # 1. Data Retrieval (Web Scraping)
    scraped_data = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status() 
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            scraped_data.append(text)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue 

    # 2. Data Synthesis (Generative AI Model)
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-001')

    def summarize_data(data, topic):
        """Summarizes the scraped data using a generative AI model."""
        prompt = f"Summarize the following information about {topic} in under 500 tokens: {data}"
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error summarizing data: {e}")
            return "Error summarizing data."

    def chunk_data(data, chunk_size=3000):
        """Chunks the data into smaller pieces."""
        chunks = []
        current_chunk = ""
        for item in data:
            if len(current_chunk) + len(item) < chunk_size:
                current_chunk += item + " "
            else:
                chunks.append(current_chunk)
                current_chunk = item + " "
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    chunked_data = chunk_data(scraped_data)
    summarized_chunks = [summarize_data(chunk, topic) for chunk in chunked_data]
    summarized_data = " ".join(summarized_chunks)

    def generate_report(data, topic):
      """Generates a report using a generative AI model."""
      prompt = f"Write a research report on {topic} using the following information: {data}. The report should be well-structured and comprehensive."
      try:
          response = model.generate_content(prompt)
          return response.text.strip()
      except Exception as e:
          print(f"Error generating report: {e}")
          return "Error generating report."

    def refine_and_organize(report, topic):
        """Refines and organizes the generated report into a logical format."""
        prompt = f"Refine, summarize, and organize the following research report on {topic} to ensure it is well-structured and logically coherent. Keep the refined report under 1500 tokens: {report}"
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error refining and organizing report: {e}")
            return "Error refining and organizing report."

    refined_report = refine_and_organize(generate_report(summarized_data, topic), topic)


    return refined_report


topic = "How AI agents are working and what are the impacts of them in technology?"
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
]

report = research_and_write(topic, urls)

print("Research Report:\n", report)