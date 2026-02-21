@echo off\r\npython -m venv venv\r\ncall venv\Scripts\activate\r\npip install -r requirements.txt\r\nstreamlit run app.py
