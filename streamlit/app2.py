import os
import chainlit as cl

# Define storage directories
PDF_STORAGE_DIR = "uploaded_pdfs"
TXT_STORAGE_DIR = "uploaded_txts"

# Ensure directories exist
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
os.makedirs(TXT_STORAGE_DIR, exist_ok=True)

@cl.on_chat_start
async def start():
    await cl.Message("📂 Please upload PDFs and TXT files separately.").send()

    # Ask for PDF files
    pdf_response = await cl.AskFileMessage(
        title="📄 Upload your PDF files",
        accept=["application/pdf"],
        max_files=5
    )

    if pdf_response:  
        uploaded_pdfs = []
        for file in pdf_response:
            file_path = os.path.join(PDF_STORAGE_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.content)  
            uploaded_pdfs.append(file.name)
        await cl.Message(f"✅ PDFs uploaded: {', '.join(uploaded_pdfs)}").send()
    else:
        await cl.Message("⚠️ No PDFs uploaded.").send()

    # Ask for TXT files
    txt_response = await cl.ask_for_files(
        title="📝 Upload your TXT files",
        accept=["text/plain"],
        max_files=5
    )

    if txt_response:  
        uploaded_txts = []
        for file in txt_response:
            file_path = os.path.join(TXT_STORAGE_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.content)  
            uploaded_txts.append(file.name)
        await cl.Message(f"✅ TXTs uploaded: {', '.join(uploaded_txts)}").send()
    else:
        await cl.Message("⚠️ No TXTs uploaded.").send()
