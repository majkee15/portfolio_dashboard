FROM python:3.7 as build_image

WORKDIR /src
COPY . /src/

RUN pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install --system && \
    pip list

EXPOSE 8050

CMD cd src

ENTRYPOINT python dashboard.py