sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
echo "Configuring shell for Pyenv..."
echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
echo "restarting shell"
source ~/.bashrc
pyenv install 3.10.12
pyenv global 3.10.12
sudo apt install git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python black_friday_pipeline/pipeline/pipeline_definition.py
python black_friday_pipeline/pipeline/pipeline_submit.py
