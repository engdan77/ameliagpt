# AmeliaGPT

More details around this project found at 
https://dtools.ipsoft.com/confluence/display/CO/AmeliaGPT

## Installation
- Install Python 3.11 or above

```shell
$ git clone https://dtools.ipsoft.com/bitbucket/scm/~dengvall/ameliagpt.git
$ cd AmeliaGPT && python3 -m pip install -r requirements.txt
$ mkdir docs   (place your docs there)
```
Note this was tested using langchain==0.0.184


- Add API keys
````shell
$ vi .env

OPENAI_API_KEY=xxx
NGROK_AUTHTOKEN=xxx
````

#### Get above API keys at
- https://openai.com/blog/openai-api
- https://ngrok.com/docs/api/resources/api-keys/

#### Give support for GPT4ALL (local LLM)
Also include following steps
```shell
# Based on https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
git clone --recurse-submodules https://github.com/nomic-ai/gpt4all.git
cd gpt4all/gpt4all-backend/
mkdir build
cd build
cmake ..
cmake --build . --parallel  # optionally append: --config Release
# Confirm that libllmodel.* exists in gpt4all-backend/build
cd ../../gpt4all-bindings/python
pip3 install -e .
```


## Run
This would use all documents found in docs, in the future there might be external API's etc.
````shell
$ python3 -m ameliagpt ./docs
````


## How to use

#### Local API docs
http://127.0.0.1:8000/api

#### Conversation tester
http://127.0.0.1:8000/conversation/


## Author
Feel free to contact daniel.engvall@amelia.com 