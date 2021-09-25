FROM tensorflow/tensorflow

RUN pip3 install --upgrade pip

RUN pip3 install  pandas

RUN pip3 install scikit-learn

COPY stock.csv /

COPY pred.py /

COPY hello.py /

CMD ["/hello.py"]

ENTRYPOINT ["python3"]

