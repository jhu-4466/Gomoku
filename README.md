# Gomoku
PyQt + Pygame, AI agent

## Clone the repository to local
```shell
git clone git@github.com:jhu-4466/Gomoku.git
```

or download from the website directly.

## Goes to your saved path, then news and starts a venv environment
```shell
python3 -m venv gomoku_env
./gomoku_env/Scripts/Activate.ps1
```

## Installs required libraries
```shell
pip install requirements.txt
```

## Goes to the code folder, then runs the program
Here you need to open at least 3 cmd windows or powershell windows or so on.
```shell
cd src
python ./gomokuapp.py
python ./negamaxagent.py
python ./aiwrapper.py -m "./source/yourGomocupAIpath/yourGomocupAI.exe"
```

if there is no "source" folder in ./src, pls new it first.
