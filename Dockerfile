FROM continuumio/anaconda3

RUN apt update && apt -y upgrade && \
    apt install -y sudo zip unzip vim nodejs npm curl wget git

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    apt install -y python3-venv python3-pip

RUN pip install --upgrade pip build twine

RUN pip install tqdm python-dotenv && \
    pip install polars pandas numpy scipy pandas pyarrow fastparquet fastexcel && \
    pip install altair vl-convert-python great_tables matplotlib matplotlib_venn japanize_matplotlib seaborn plotly
    