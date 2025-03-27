import streamlit as st
import os
import numpy as np
import PyPDF2

from together import Together
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

# API Keys (Replace with your actual keys)
TOGETHER_API_KEY = "8b41e536935c171ab7eef4bfe5e9dea15fe5a105277fc2b49f8c1e389678a319"
PINECONE_API_KEY = "pcsk_7VH2Wt_M8DP8vcE3vJ9cMMTTKFrsBFUc4PimQSBYE416cHz5nPgxD6eSYVoTJ3PXLBZW5"
# Initialize clients
client = Together(api_key=TOGETHER_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mooligai"
time.sleep(1)
index = pc.Index(index_name)
time.sleep(1)

class MooligAI:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.web_searcher = DDGS()
        self.pdf_vectorstore = None

    def extract_pdf_text(self, pdf_file):
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""

    def create_pdf_vectorstore(self, pdf_text):
        """Create vector store from PDF text"""
        try:
            # Split text into chunks
            text_chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
            
            # Create vector store
            self.pdf_vectorstore = FAISS.from_texts(
                text_chunks, 
                self.embeddings
            )
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False

    def get_pdf_context(self, query):
        """Retrieve context from PDF vector store"""
        if not self.pdf_vectorstore:
            return "No PDF loaded. Please upload a PDF first."
        
        try:
            # Retrieve similar documents
            similar_docs = self.pdf_vectorstore.similarity_search(query, k=3)
            context = " ".join([doc.page_content for doc in similar_docs])
            return context
        except Exception as e:
            st.error(f"Error retrieving PDF context: {e}")
            return "Error retrieving context from PDF."

    def get_context(self, query):
        """Retrieve context from Pinecone vector database"""
        query_embeddings = self.embeddings.embed_query(query)
        embedding_numpy = np.array(query_embeddings)
        retrieval_data = index.query(
            namespace="ayurveda",
            vector=embedding_numpy.tolist(),
            top_k=5,
            include_values=False,
            include_metadata=True
        )

        context_list = [
            match.get("metadata", {}).get("source_text", "").strip()
            for match in retrieval_data.get("matches", [])
            if match.get("metadata", {}).get("source_text", "").strip()
        ]

        return " ".join(context_list) if context_list else "No relevant Ayurvedic context found."

    def web_search(self, query, num_results=3):
        """Perform web search using DuckDuckGo"""
        search_results = list(self.web_searcher.text(query, max_results=num_results))
        return "\n\n".join([
            f"🌐 **Source {i+1}:** {result['title']}\n🔗 {result['href']}\n📖 {result['body']}"
            for i, result in enumerate(search_results)
        ])

    def context_validation(self, query):
        """
        Validate if current context is sufficient to answer the query
        If not, generate additional context retrieval query
        """
        # Retrieve existing contexts
        pinecone_context = self.get_context(query)
        pdf_context = self.get_pdf_context(query) if self.pdf_vectorstore else ""
        
        # Context validation prompt
        validation_prompt = f"""
        You are an expert knowledge assessor. Evaluate the following contexts 
        to determine if they provide sufficient information to comprehensively 
        answer the user's query:

        User Query: {query}

        Pinecone Context:
        {pinecone_context}

        PDF Context:
        {pdf_context}

        Assessment Task:
        1. Determine if the current contexts fully address the query
        2. If contexts are insufficient, generate:
           a) A list of specific information gaps
           b) Precise web search queries to fill those gaps
        3. Recommend whether to:
           - Proceed with current context
           - Perform additional web searches
           - Modify the query

        Response Format:
        - Sufficiency Rating: [Full/Partial/Insufficient]
        - Information Gaps (if any):
        - Recommended Web Search Queries (if applicable):
        - Recommendation: [Use Current Context/Perform Web Search/Refine Query]
        """

        # Generate validation
        validation = self.generate_response(validation_prompt)
        return validation, pinecone_context, pdf_context

    def generate_comprehensive_response(self, query, pinecone_context, pdf_context, validation_result):
        """
        Generate a comprehensive response based on context validation
        """
        # Parse validation result to determine next steps
        web_search_needed = "Perform Web Search" in validation_result

        # Prepare web search context if needed
        web_context = ""
        web_used = False
        if web_search_needed:
            # Extract recommended search queries from validation
            search_query_match = re.search(r'Recommended Web Search Queries:(.*?)Recommendation:', validation_result, re.DOTALL)
            if search_query_match:
                search_queries = search_query_match.group(1).strip().split('\n')
                for query in search_queries:
                    query = query.strip('- ')
                    if query:
                        web_context += self.web_search(query) + "\n\n"
                web_used = True

        # Combine all contexts
        combined_context = f"""
        Pinecone Context:
        {pinecone_context}

        PDF Context:
        {pdf_context}

        Web Context (if used):
        {web_context}
        """

        # Generate final response prompt
        response_prompt = f"""
        You are Moolig.AI, an expert in Ayurveda. 
        Provide a comprehensive, accurate answer to the following query:

        User Query: {query}

        Available Contexts:
        {combined_context}

        Response Guidelines:
        - Base your answer on the available contexts
        - Be clear and concise
        - Provide actionable insights
        - Reference sources when possible
        - Indicate if web sources were used to supplement the answer
        """

        # Generate final response
        final_response = self.generate_response(response_prompt)

        # Append web usage note
        if web_used:
            final_response += "\n\n*Note: Additional information was sourced from web searches to provide a comprehensive answer.*"
        else:
            final_response += "\n\n*Note: Answer based entirely on existing knowledge bases.*"

        return final_response

    def generate_response(self, prompt):
        """Generate response using Together AI"""
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I apologize, but I couldn't generate a response at the moment."

def main():
    # Custom Styling (Previous styling remains the same)
    st.markdown("""
    <style>
        body { background-color: #F5F1E3; font-family: 'Arial', sans-serif; }
        .title { 
            text-align: center; 
            color: #4A7856; 
            font-size: 40px; 
            font-weight: bold; 
            margin: 10px 0;
        }
        /* ... other styles remain the same ... */
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.image("https://via.placeholder.com/150", caption="Moolig.AI", use_column_width=True)
    
    # Initialize RAG System
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MooligAI()

    # Sidebar PDF Upload (Previous PDF upload remains the same)
    st.sidebar.header("📚 PDF Knowledge Base")
    uploaded_pdf = st.sidebar.file_uploader("Upload Ayurveda PDF", type=['pdf'])
    
    if uploaded_pdf is not None:
        # Extract and process PDF
        with st.sidebar.status("Processing PDF..."):
            pdf_text = st.session_state.rag_system.extract_pdf_text(uploaded_pdf)
            if pdf_text:
                if st.session_state.rag_system.create_pdf_vectorstore(pdf_text):
                    st.sidebar.success("PDF Successfully Processed!")
            else:
                st.sidebar.error("Failed to process PDF")

    # Main App Title
    st.markdown("""
    <div class='leaf-bg'>
        <p class='title'>🌿 Moolig.AI 2.0 🌿</p>
        <p class='title'>Advanced Ayurveda Knowledge Hub</p>
    </div>
    """, unsafe_allow_html=True)

    # User Query Section
    prompt1 = st.text_input("🌿 Ask about Ayurveda:", placeholder="Enter your holistic health question...")

    # Query Processing
    if prompt1:
        with st.spinner("Processing your query..."):
            # Context Validation
            validation_result, pinecone_context, pdf_context = st.session_state.rag_system.context_validation(prompt1)
            
            # Display Validation Details
            with st.expander("🔍 Context Validation"):
                st.markdown(validation_result)
            
            # Generate Comprehensive Response
            comprehensive_answer = st.session_state.rag_system.generate_comprehensive_response(
                prompt1, 
                pinecone_context, 
                pdf_context, 
                validation_result
            )
            
            # Display Answer
            st.markdown("### 🌱 Moolig.AI Response")
            st.markdown(comprehensive_answer)

    # Web Search Option
    if prompt1:
        with st.expander("🌐 Web Search Results"):
            web_results = st.session_state.rag_system.web_search(prompt1)
            st.markdown(web_results, unsafe_allow_html=True)

if __name__ == "__main__":
    main()