FROM nvidia/cudagl:10.1-devel-ubuntu16.04

#****************Setting up user *********************
ARG UNAME=ros
ARG PASS=ros
ARG UID=1000
ARG GID=1000
RUN apt-get update && apt-get install sudo -y
RUN sudo -i
RUN groupadd -g $GID -o $UNAME
RUN sudo useradd -m -u $UID -g $GID -p $(openssl passwd -1 $PASS) -o -s  /bin/bash $UNAME
RUN usermod -aG sudo $UNAME
USER $UNAME 
WORKDIR home/$UNAME/ 

#********************Installing Extra packaged**********************
RUN echo $PASS | sudo -S echo "fuck me!"
RUN echo $PASS | sudo -S apt-get update
RUN echo $PASS | sudo -S apt-get install -y software-properties-common
#gedit
RUN echo "Y" |sudo -S apt-get install gedit
#python2.7
RUN echo "gedit"&& echo $PASS | sudo -S apt-get update
RUN echo "gedit"&& echo "Y" | sudo -S apt upgrade
RUN echo "python"&& echo "Y" | sudo -S apt install python2.7 python-pip
#python 3.6
# RUN echo "\u000A" |  sudo -S add-apt-repository ppa:jonathonf/python-3.6 
# RUN  sudo -S apt-get update
# RUN echo "Y"|sudo -S apt-get install python3.6
# RUN sudo -S update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
# RUN sudo -S update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
# RUN sudo -S update-alternatives --config python3
# RUN sudo -S apt-get -y install python3-pip
# RUN echo "Y"|sudo -S apt-get remove python3-apt
# RUN echo "Y"|sudo -S apt-get install python3-apt
#installing terminator
    #"\u000A" is like pressing the Enter key
RUN echo $PASS | sudo -S apt-get install -y software-properties-common
RUN echo "\u000A" | sudo -S add-apt-repository ppa:gnome-terminator
RUN echo $PASS | sudo -S apt-get update
RUN echo "Y" | sudo -S apt-get install terminator

#*********************************************************************
CMD /bin/bash
