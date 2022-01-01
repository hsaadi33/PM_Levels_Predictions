FROM python:3.9

COPY requirements.txt .
COPY . /home/PM_levels_project

RUN pip install -r requirements.txt

WORKDIR /home/PM_levels_project

CMD ["bash"]