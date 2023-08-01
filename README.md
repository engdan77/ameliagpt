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

- Add API keys
````shell
$ vi .env

OPENAI_API_KEY=xxx
NGROK_AUTHTOKEN=xxx
````

# Get above API keys at
- https://openai.com/blog/openai-api
- https://ngrok.com/docs/api/resources/api-keys/


## Run

````shell
$ python3 -m ameliagpt
````


## How to use

### Local API docs
http://127.0.0.1:8000/api

### Conversation tester
http://127.0.0.1:8000/conversation/


## Author
Feel free to contact daniel.engvall@amelia.com 