from langchain_core.prompts import PromptTemplate

# ✅ Corrected Prompt Template to Accept Context
system_prompt = PromptTemplate(
    input_variables=["context", "input"],  # 🔹 Ensure BOTH `context` and `input` are recognized
    template="""You are a knowledgeable medical assistant. 
Use the provided medical database context to answer user queries.

Context: {context}

User Query: {input}

If no relevant information is found, suggest consulting a healthcare professional."""
)
