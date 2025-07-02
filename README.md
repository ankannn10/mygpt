# MyGPT — Gemini ChatGPT Clone

**MyGPT** is a multi-session chatbot interface built with **Streamlit** and powered by **Google's Gemini 1.5 Pro API**.
It supports contextual conversations, auto-generated titles, PDF exports, and a clean UI for a ChatGPT-style experience.

---

## Features

* Multi-session chat interface with memory per chat
* Auto-generated chat titles using Gemini
* Manual renaming and deletion of chat sessions
* Export individual chats to PDF with styled formatting
* Full contextual understanding using Gemini's chat history
* Streamlit-powered interactive UI

---

## Getting Started

Follow the steps below to set up and run the app locally.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mygpt.git
cd mygpt
```

### 2. Create and activate a virtual environment

**For macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows (CMD):**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key

You can set your API key using one of the following methods:

#### Option A — Export it in your terminal:

**macOS/Linux:**

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

**Windows CMD:**

```cmd
set GOOGLE_API_KEY=your-api-key-here
```

#### Option B — (Optional) Create a `.env` file:

```
GOOGLE_API_KEY=your-api-key-here
```

You can get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

---

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to use the chatbot.

---

## Requirements

* Python 3.8 or higher
* A valid Gemini API key
* Internet connection to access the Gemini API

---

## Future Enhancements (Planned)

* Dark mode toggle
* Search functionality across chats
* Image input for multimodal Gemini support
* Feedback/rating for responses
* Multi-user login support

---

## License

This project is open-source and available under the MIT License.
