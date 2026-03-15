# System Architecture

This document describes the architecture of the Airbnb Streamlit Dashboard project.

The goal of this document is to help AI coding agents and developers quickly understand how the system works.

---

# System Overview

The application is a **Streamlit-based analytics dashboard** for exploring Airbnb listing data.

The application consists of three main layers:

1. Authentication Layer
2. Data Processing Layer
3. Visualization / Dashboard Layer

Users must authenticate before accessing the dashboard.

---

# High Level Architecture

User
   │
   ▼
Authentication System (users.py)
   │
   ▼
Streamlit Application (app.py)
   │
   ├── Data Loading
   ├── Data Preprocessing
   ├── Visualization
   └── Dashboard Pages
   │
   ▼
Dataset (Airbnb_Open_Data.csv)

---

# Component Description

## 1 Authentication Layer

File:
users.py

Responsibilities:

- Register new users
- Login existing users
- Logout users
- Manage authentication session

Authentication state is stored in:

st.session_state["authenticated"]

Optional session data:

st.session_state["username"]

If authentication fails, users are redirected to the login page.

---

# 2 Streamlit Application Layer

File:
app.py

Responsibilities:

- Render Streamlit UI
- Control page navigation
- Load dataset
- Generate visualizations
- Display insights

The application contains multiple sections (pages).

---

# 3 Dashboard Pages

The application contains the following pages:

Authentication
    Register
    Login

Dashboard

    Overview
    Data Raw
    Preprocessing
    EDA Insights
    Conclusion
    Chatbot

Each section is rendered using Streamlit components.

---

# Data Flow

The typical data flow is:

1. User logs in
2. Streamlit loads dataset
3. Data preprocessing is performed
4. Visualizations are generated
5. Insights are displayed on dashboard

Flow diagram:

User Login
     │
     ▼
Load Dataset
     │
     ▼
Data Cleaning
     │
     ▼
EDA Analysis
     │
     ▼
Visualization
     │
     ▼
Dashboard Insights

---

# Dataset

Dataset used:

Airbnb_Open_Data.csv

Key columns include:

- price
- neighbourhood_group
- room_type
- reviews
- availability

---

# Visualization System

Charts are generated using:

Plotly

Common visualizations:

- Price distribution
- Room type comparison
- Neighborhood analysis
- Review trends

---

# UI Layout

The dashboard uses Streamlit layout components:

st.sidebar
st.columns
st.container

The UI should remain:

- clean
- readable
- minimal

---

# Authentication Flow

User → Register Account
User → Login
System → Create session

If session authenticated:

Access dashboard

Else:

Redirect to login page.

---

# Future Architecture (Optional)

If the project grows, it may be refactored into modules:

modules/

    overview.py
    data_raw.py
    preprocessing.py
    eda.py
    conclusion.py
    chatbot.py

Current version keeps most logic inside app.py.

---

# Architecture Goals

The system architecture aims to:

- Keep the code simple
- Maintain readability
- Enable fast development
- Support AI coding agents

The architecture prioritizes clarity over complexity.