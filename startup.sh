sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
pyenv --version
pyenv install 3.10.12
pyenv global 3.10.12
sudo apt install git
git clone https://github.com/data-max-hq/gcp-mlspecialization-demo2.git
cd gcp-mlspecialization-demo2
python3 -m venv venv
source venv/bin/activate
pip install requirements.txt
cd black_friday_pipeline
python -m pipeline/run_pipeline