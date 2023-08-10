import streamlit as st
import smtplib
import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings , HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from htmlTemplates import css , bot_template ,user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def send_email(subject, message, from_email, to_emails, smtp_server, smtp_port, smtp_username, smtp_password):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_emails, msg.as_string())


def generate_offer_letter(name, post):
    # Offer letter template with placeholders for name and post
    offer_letter_template = f"""
    Date: [DATE]
    [Name]
    
    Dear [Name],
    We are pleased to offer you the position of [Post] at our company.

    Please feel free to contact us if you have any questions or concerns.

    Best regards,
    AXIS BANK
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # Replace placeholders with user-provided values
    offer_letter = offer_letter_template.replace("[DATE]", formatted_datetime)  # Replace with the current date
    offer_letter = offer_letter.replace("[Name]", name)
    offer_letter = offer_letter.replace("[Post]", post)

    return offer_letter


def main():
    load_dotenv()
    st.set_page_config(page_title="CV shorlisting",
                       page_icon="📃")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple CVs 📃")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your CVs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        smtp_servers = {
            "Gmail": {
                "server": "smtp.gmail.com",
                "port": 587
            },
            "Outlook.com / Office 365": {
                "server": "smtp.office365.com",
                "port": 587
            },
            "Yahoo Mail": {
                "server": "smtp.mail.yahoo.com",
                "port": 587
            },
            "AOL Mail": {
                "server": "smtp.aol.com",
                "port": 587
            }
        }

        # Adding email functionality with dropdown for SMTP servers
        st.sidebar.subheader("Email")
        st.sidebar.write("Enter email addresses separated by commas (',')")
        to_emails = st.sidebar.text_input("To:")
        subject = st.sidebar.text_input("Subject:")
        message = st.sidebar.text_area("Message:")
        selected_server = st.sidebar.selectbox("Select SMTP Server:", list(smtp_servers.keys()))
        smtp_server = smtp_servers[selected_server]["server"]
        smtp_port = smtp_servers[selected_server]["port"]
        smtp_username = st.sidebar.text_input("Enter Username:")
        smtp_password = st.sidebar.text_input("Enter Password:", type="password")

        if st.sidebar.button("Send"):
            if to_emails and subject and message and smtp_server and smtp_username and smtp_password:
                to_emails = [email.strip() for email in to_emails.split(",")]
                send_email(subject, message, smtp_username, to_emails, smtp_server, smtp_port, smtp_username,
                           smtp_password)
                st.sidebar.success("Email sent successfully!")
            else:
                st.sidebar.warning("Please fill in all the email details.")

        # Adding offer letter generation feature
        st.sidebar.subheader("Offer Letter Generation")
        name = st.sidebar.text_input("Enter Name:")
        post = st.sidebar.text_input("Enter Post:")
        if st.sidebar.button("Generate Offer Letter"):
            if name and post:
                offer_letter = generate_offer_letter(name, post)
                st.sidebar.success("Offer Letter generated successfully!")
                st.sidebar.text_area("Offer Letter:", offer_letter, height=300)
            else:
                st.sidebar.warning("Please enter both Name and Post to generate the Offer Letter.")


if __name__ == '__main__':
    main()