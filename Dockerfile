FROM python:3.11

WORKDIR /mini-gpt-like-lm

COPY . /mini-gpt-like-lm

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8000 8501

ENTRYPOINT [ "minigpt" ]