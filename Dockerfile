#FROM nvcr.io/nvidia/pytorch:21.12-py3
FROM nvcr.io/nvidia/pytorch:22.04-py3
#FROM nvcr.io/nvidia/pytorch:22.07-py3

ARG proxy
ARG jupyter_password

ENV METAL_HOSTS=10.79.85.55
ENV HTTP_PROXY=$proxy
ENV HTTPS_PROXY=$proxy
ENV http_proxy=$proxy
ENV https_proxy=$proxy
ENV no_proxy=localhost,$METAL_HOSTS,127.0.0.0,127.0.1.1,127.0.1.1,local.home,ge.com,bakerhughes.com
ENV NO_PROXY=$no_proxy
ENV MLFLOW_TRACKING_URI=http://10.79.85.55:5000
ENV NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/GEExternalRootCA2.1.crt

COPY *.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN echo $proxy
RUN echo $jupyter_password

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx && apt-get clean



# PYTHON
RUN pip3 install --upgrade opencv-python &&\
    pip3 install --upgrade scikit-image seaborn &&\
    pip3 install shap umap-learn

# PYTORCH-GEOMETRIC DEPENDENCIES MUST BE INSTALLED FROM WHEELS, RATHER THAN FROM PIP, OTHERWISE
# INSTALLATION IS FROM SOURCE & VERY SLOW. REMEMBER TO UPDATE THE TORCH & CUDA ENV VARIABLES
# IF YOU CHANGE THE BASE IMAGE!!!!
RUN TORCH=$(python -c "import torch; print(torch.__version__)"); \
    CUDA=$(python -c "import torch; print(torch.version.cuda)" | sed 's/\.//g'); \
    pip install -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html torch-scatter torch-sparse
RUN pip install torch-geometric torch-geometric-temporal pytorch-lightning

# JUPYTER
#RUN pip3 install --upgrade jupyter jupyterlab jupyterlab-git xeus-python

## TENSORBOARD
RUN pip3 install jupyter_tensorboard 
RUN jupyter tensorboard enable

RUN jupyter notebook --generate-config && echo "c.NotebookApp.password='$jupyter_password'" >> /root/.jupyter/jupyter_notebook_config.py &&\  
    echo "c.NotebookApp.ip = '*'" >>/root/.jupyter/jupyter_notebook_config.py &&\ 
    echo "c.NotebookApp.open_browser = False" >>/root/.jupyter/jupyter_notebook_config.py &&\ 
    echo "c.NotebookApp.port = 8888" >>/root/.jupyter/jupyter_notebook_config.py

#RUN    jupyter contrib nbextension install && jupyter nbextensions_configurator enable

RUN pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install && jupyter nbextension enable varInspector/main

# SSH SERVER
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ailab 
#RUN echo 'ailab:00000000' | chpasswd
RUN echo 'root:BH.restricted17.ssh' | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

WORKDIR /

RUN echo "#!/bin/bash" >> /root/start_jupyter.sh
RUN echo "export no_proxy=$no_proxy" >> /root/start_jupyter.sh
RUN echo "export NO_PROXY=$no_proxy" >> /root/start_jupyter.sh
RUN echo "env | grep _ >> /etc/environment" >> /root/start_jupyter.sh
RUN echo "service ssh start" >> /root/start_jupyter.sh
RUN echo "jupyter notebook --allow-root --ip=0.0.0.0" >> /root/start_jupyter.sh
RUN chmod +x /root/start_jupyter.sh

EXPOSE 22
EXPOSE 8888 5000 8080
CMD ["/root/start_jupyter.sh"]
