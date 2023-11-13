FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY Dockerfile ./Dockerfile
COPY get_around_delay_analysis.xlsx ./get_around_delay_analysis.xlsx
COPY streamlit_app.py ./streamlit_app.py

CMD streamlit run --server.port $PORT streamlit_app.py