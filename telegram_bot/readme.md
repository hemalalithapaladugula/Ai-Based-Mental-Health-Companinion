
steps to run the code :

1. open the folder in VS Code
2. open new terminal
3. create new environment : command : 
4. for windows : 

```
python -m venv venv
```

1. activate the environment : for windows : 

```
.\\venv\\Scripts\\activate   
```

if this dont work, try 

```
venv\Scripts\activate
```

1. install the dependencies with this command : 

```
pip install -r requirements.txt
```

1. after installing, run the code : command : 

```
uvicorn mail:app --reload
```

1. go to telegram and test