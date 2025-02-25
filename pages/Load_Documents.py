import os
import openai
from utils._admin_util import create_embeddings, create_vector_store, read_pdf_data, split_data
import streamlit as st
from dotenv import load_dotenv


def main():
    load_dotenv()
    
    # Add detailed API key verification
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found! Please ensure it's set in the environment variables.")
        st.info("To set up your API key:")
        st.code("1. Go to Hugging Face Space settings\n2. Add OPENAI_API_KEY in Repository Secrets")
        st.stop()
    

    st.set_page_config(page_title="Dump PDFs to QDrant - Vector Store")
    st.title("Please upload your files...📁 ")
    try:
        # Upload multiple PDF files
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
        
            with st.spinner('Processing PDF files...'):
                all_chunks = []
                
                # Process each PDF file
                for pdf in uploaded_files:
                  
                    st.write(f"Processing: {pdf.name}")
                    
                    # Extract text from PDF
                    text = read_pdf_data(pdf)
                    st.write(f"👉 Reading {pdf.name} done")

                    # Create chunks for this PDF
                    chunks = split_data(text)
                    all_chunks.extend(chunks)
                    st.write(f"👉 Splitting {pdf.name} into chunks done")
                    
                if not all_chunks:
                    st.error("❌ No valid chunks were created from the PDFs")
                    st.stop()

                st.write("Creating embeddings...")
                embeddings = create_embeddings()
                st.write("👉 Creating embeddings instance done")
        
                # Create vector store with all chunks
                vector_store = create_vector_store(embeddings, all_chunks)
                st.session_state.vector_store = vector_store
            
                st.success(f"✅ Successfully processed {len(uploaded_files)} files and pushed embeddings to Qdrant")
                st.write(f"Total chunks created: {len(all_chunks)}")

    except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            
            
if __name__ == '__main__':
    main()
