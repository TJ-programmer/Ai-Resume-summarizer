````markdown name=README.md
# 🤖 AI Resume Summarizer Agent

**AI Resume Summarizer Agent** is a modern, interactive Streamlit application that leverages the power of LLMs, RAG (Retrieval-Augmented Generation), and CrewAI agents to analyze, summarize, and provide actionable insights on resumes. Upload your resume (PDF, DOCX, or TXT), get a detailed AI-driven analysis, chat with your resume, and download a comprehensive report!

---

## 🚀 Features

- **Multi-Format Support:** Upload resumes in PDF, DOCX, or TXT formats.
- **AI-Powered Analysis:** CrewAI agents conduct deep analysis across resume content, skills, and market fit.
- **Conversational RAG Chat:** Ask questions about your resume and get context-aware answers.
- **Beautiful PDF Report:** Download a structured, easy-to-read PDF report with all analysis and chat history.
- **Rich UI:** Clean, modern Streamlit interface with an intuitive workflow.
- **Secure:** API keys are handled securely; resume data is processed in-memory.

---

## 🖥️ Demo

![AI Resume Summarizer Demo](assets/demo.gif)

---

## 🧠 How It Works

1. **Upload:** Choose your resume file (PDF, DOCX, TXT).
2. **Extract & Embed:** The app extracts text and creates a semantic vector store for search and RAG chat.
3. **Analyze:** CrewAI agents (Senior Resume Analyst, Skills Evaluator, Market Researcher) analyze your resume in parallel.
4. **Chat:** Ask questions about your resume content, get AI-powered answers.
5. **Report:** Download a detailed, well-formatted PDF containing all analyses and your chat Q&A.

---

## ✨ Example Output

- **Resume Analysis:** Detailed strengths, weaknesses, and professional summary
- **Skills Breakdown:** Technical, soft, industry-specific skill evaluation
- **Market Research:** Salary benchmarks, target roles, market trends
- **Chat History:** Every question and answer, all in one report!

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for the interactive UI
- [LangChain](https://python.langchain.com/) for text processing, embedding, RAG
- [CrewAI](https://github.com/joaomdmoura/crewai) for collaborative agent-based analysis
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [PyPDF2](https://pypi.org/project/PyPDF2/), [python-docx](https://python-docx.readthedocs.io/), [ReportLab](https://www.reportlab.com/) for file handling and PDF generation
- **LLMs:** [Groq](https://groq.com/) (Llama3, etc.)

---

## ⚡ Quickstart

### 1. **Clone the Repo**

```bash
git clone https://github.com/TJ-programmer/Ai-Resume-summarizer.git
cd Ai-Resume-summarizer
```

### 2. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 3. **Set Up API Keys**

- Create a `.env` file or set the environment variable `GROQ_API_KEY` with your [Groq API key](https://console.groq.com/).
- Or, paste the key in the app sidebar at runtime.

### 4. **Run the App**

```bash
streamlit run app.py
```

---

## 📝 Usage

1. **Enter your Groq API Key** in the sidebar.
2. **Upload your resume** (PDF, DOCX, or TXT).
3. **Click 'Generate AI Analysis'** to run CrewAI agents.
4. **Chat** with your resume to get custom AI responses.
5. **Download your PDF report** with one click.

---

## 🧩 Project Structure

```
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── assets/               # Images, demo gifs, etc. (optional)
└── README.md             # This file!
```

---

## 🤝 Contributing

Pull requests welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or PR.


---

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewai)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [Groq](https://groq.com/)

---

## 💡 Ideas for Improvement

- Support for more file formats (e.g., LinkedIn exports)
- Integration with job boards for real-time targeting
- Customizable report templates
- Multilingual support

---

<div align="center">
  <b>Level up your resume with AI! 🚀</b>
</div>
````
