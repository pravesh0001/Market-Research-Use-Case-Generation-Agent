import streamlit as st
import time
from duckduckgo_search import DDGS
from transformers import pipeline


import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Market Research & AI Use Case Generator", layout="wide")


@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    generator = pipeline('text-generation', model='gpt2')
    return summarizer, generator

summarizer, generator = load_models()



class ResearchAgent:
    def __init__(self, company):
        self.company = company

    def search_info(self):
        query = f"{self.company} industry overview"
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                texts = [r['body'] for r in results if 'body' in r]
                if not texts:
                    return "No relevant research data found."
                full_text = " ".join(texts)
                return full_text
        except Exception as e:
            return f"Error fetching research data: {e}"

    def summarize(self, text):
        if not text:
            return "No data found."
        summarized = summarizer(text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
        return summarized

class UseCaseGeneratorAgent:
    def __init__(self, research_summary):
        self.research_summary = research_summary

    def generate_use_cases(self):
        prompt = f"""Given this research:

{self.research_summary}

Suggest 5 relevant AI/ML/GenAI use cases to improve operations, customer experience, or efficiency.
"""
        generated = generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
        lines = generated.split('\n')
        use_cases = [line.strip("- ").strip() for line in lines if line.strip()]
        return use_cases[:5]

class DatasetCollectorAgent:
    def __init__(self, use_cases):
        self.use_cases = use_cases

    def search_datasets(self):
        datasets = {}
        for case in self.use_cases:
            query = f"{case} dataset site:kaggle.com OR site:huggingface.co"
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=2)
                    links = [r['href'] for r in results if 'href' in r]
                    if not links:
                        datasets[case] = ["No datasets found."]
                    else:
                        datasets[case] = links
                time.sleep(1)  # Respect API rate limits
            except Exception as e:
                datasets[case] = [f"Error fetching datasets: {e}"]
        return datasets



st.title("📊 Market Research & AI Use Case Generation Agent")

company_name = st.text_input("Enter Company or Industry Name:", "")

if st.button("Run Research"):
    if company_name:
        with st.spinner("Researching Industry..."):
            researcher = ResearchAgent(company_name)
            full_info = researcher.search_info()
            summary = researcher.summarize(full_info)

        st.subheader("🔎 Research Summary")
        st.write(summary)

        with st.spinner("Generating AI Use Cases..."):
            usecase_agent = UseCaseGeneratorAgent(summary)
            use_cases = usecase_agent.generate_use_cases()

        st.subheader("🚀 AI Use Cases Suggested")
        for idx, case in enumerate(use_cases, 1):
            st.write(f"{idx}. {case}")

        with st.spinner("Searching Resource Datasets..."):
            dataset_agent = DatasetCollectorAgent(use_cases)
            dataset_links = dataset_agent.search_datasets()

        st.subheader("📚 Dataset Resources")
        for uc, links in dataset_links.items():
            st.write(f"**{uc}**")
            for link in links:
                st.markdown(f"- [Dataset Link]({link})")
            st.markdown("---")

        st.success("✅ Process Completed!")

    else:
        st.warning("Please enter a valid company or industry name.")
