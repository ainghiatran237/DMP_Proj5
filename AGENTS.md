# AGENTS.md

This file provides instructions for AI coding agents (Codex CLI, Cursor, Copilot, etc.) working in this repository.

---

# Project Overview

This project is a **Streamlit dashboard for Airbnb data analysis**.

The purpose of this project is to explore Airbnb listing data and generate insights using an interactive Streamlit dashboard.

The dashboard follows a **PowerBI-style layout** and includes data exploration, preprocessing, EDA visualization, and insights.

Users must **register and login before accessing the dashboard**.

---

# Tech Stack

Language
- Python 3+

Libraries
- Streamlit → web UI framework
- Pandas → data processing
- Plotly → interactive visualization

Dataset
- Airbnb_Open_Data.csv

---

# Current Project Structure

The project currently has a **simple structure and is not fully modularized yet**.

Files:

app.py  
→ main Streamlit application  
→ contains UI layout, dashboard pages, and visualization logic

users.py  
→ authentication logic  
→ register, login, logout, and session management

data/  
→ dataset folder containing Airbnb_Open_Data.csv

---

# Authentication System

The application includes a **user authentication system**.

Users must create an account before accessing the dashboard.

Authentication features:

- Register → create a new account
- Login → authenticate user
- Logout → end session

Authentication state is stored using Streamlit session:

st.session_state["authenticated"]

Optional session variables:

st.session_state["username"]

If a user is not authenticated, the application should redirect to the login page.

---

# Application Pages

The Streamlit dashboard contains the following sections:

1. Authentication
   - Register page
   - Login page
   - Logout

2. Overview
   - summary statistics
   - dataset overview

3. Data Raw
   - display raw Airbnb dataset
   - basic filtering

4. Preprocessing
   - data cleaning
   - missing value handling
   - feature preparation

5. EDA Insights
   - charts and visualizations
   - price analysis
   - room type distribution
   - neighborhood analysis

6. Conclusion
   - summarized insights from analysis

7. Chatbot
   - AI assistant that explains the dataset insights

---

# Data Handling

Dataset file:

Airbnb_Open_Data.csv

Important variables:

- price
- neighbourhood_group
- room_type
- reviews
- availability

Preprocessing tasks may include:

- removing invalid rows
- handling missing values
- converting price to numeric format

---

# Coding Guidelines

Because the project currently uses a simple structure:

- Most logic remains inside `app.py`
- Avoid unnecessary file creation
- Keep code readable and organized

General rules:

- Use small reusable functions
- Separate authentication logic in `users.py`
- Avoid writing extremely long functions
- Maintain clear variable names

---

# UI Guidelines

The Streamlit UI should remain:

- simple
- clean
- easy to navigate

Recommended layout components:

- st.sidebar
- st.columns
- st.container

Dashboard should resemble **modern analytics dashboards**.

---

# Visualization Guidelines

Use Plotly charts whenever possible.

Common analysis topics:

- price distribution
- neighborhood comparison
- room type analysis
- review analysis

Charts should focus on **business insights**, not only raw data.

---

# Commands

Run the application:

streamlit run app.py

Install dependencies:

pip install -r requirements.txt

---

# Agent Behavior Guidelines

When modifying this repository:

- Do not break the Streamlit layout
- Maintain readability of `app.py`
- Prefer improving existing code rather than restructuring everything
- Keep UI clean and consistent
- Avoid adding unnecessary dependencies

---

# Future Refactoring (Optional)

In the future the project may be refactored into modules:

modules/
    overview.py
    data_raw.py
    preprocessing.py
    eda.py
    conclusion.py
    chatbot.py

However this structure **does not exist yet** and should only be implemented if the project grows significantly.

---

# Project Goal

Build a **clean and interactive Airbnb analytics dashboard** that demonstrates:

- data preprocessing
- exploratory data analysis (EDA)
- business insights
- interactive visualizations
- authentication system