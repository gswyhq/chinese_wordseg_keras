FROM tensorflow/tensorflow:1.12.0-py3

# 修改时区
RUN echo "Asia/shanghai" > /etc/timezone;

ENV LANG C.UTF-8

COPY . /chinese_wordseg_keras

WORKDIR /chinese_wordseg_keras

RUN apt-get update && apt-get install -y git

RUN pip3 install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

RUN git config --global core.quotepath false && \
    git config --global log.date iso8601 && \
    git config --global credential.helper store

RUN echo 'if [ "$color_prompt" = yes ]; then' >> ~/.bashrc
RUN echo "    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
RUN echo "else" >> ~/.bashrc
RUN echo "    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '" >> ~/.bashrc
RUN echo "fi" >> ~/.bashrc

RUN echo "alias date='date +\"%Y-%m-%d %H:%M:%S\"'" >> ~/.bashrc
RUN echo "export TERM=xterm" >> ~/.bashrc
RUN echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc
RUN echo "export PATH=/bin/bash:$PATH" >> ~/.bashrc

EXPOSE 9999

CMD [ "python3"]

# $ docker build -t gswyhq/chinese_wordseg_keras -f Dockerfile .
# $ docker run -it --rm --name my-running-app gswyhq/chinese_wordseg_keras

