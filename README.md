# 🧠 Streamlit Login App

A simple Streamlit web application with user authentication.

---

## 📦 Features

- 🔐 User login system  
- 📄 Streamlit UI  
- ✅ Dependency management with `requirements.txt`  
- 💻 Cross-platform setup instructions (Windows / macOS / Linux)

---

## 🛠️ Prerequisites

- Python 3.7+
- `pip` (Python package installer)
- Optional: `virtualenv` (for managing isolated environments)

---

## 🚀 How to Run the Project

### ▶️ Option 1: Run Directly (Without Virtual Environment)

<details>
<summary>🪟 Windows</summary>

```bash
python -m pip install -r requirements.txt
streamlit run login.py
```
</details>

<details>
<summary>🍏 macOS / 🐧 Linux</summary>

```bash
python3 -m pip install -r requirements.txt
streamlit run login.py
```
</details>

---

### ▶️ Option 2: Using a Virtual Environment (Recommended)

<details>
<summary>🪟 Windows</summary>

```bash
python -m venv myenv  # Replace 'myenv' with any environment name you prefer
myenv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run login.py
```
</details>

<details>
<summary>🍏 macOS / 🐧 Linux</summary>

```bash
# Replace 'myenv' with your preferred virtual environment name if desired
python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run login.py
```
</details>