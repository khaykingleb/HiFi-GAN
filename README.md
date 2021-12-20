# Neural-Vocoder

## Model

[HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf) Implementation for TTS.


### Generator 

<p>
    <img src="img/generator.png">
</p>

### Discriminator 

<p>
    <img src="img/discriminator.png">
</p>


## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khaykingleb/Neural-Vocoder/blob/main/notebooks/notebook.ipynb)

## Getting Started

Clone the repository and step into it:

```shell
git clone https://github.com/khaykingleb/Text-to-Speech.git
cd Text-to-Speech
```

Install requirements and modules.

```shell
pip install -r requirements.txt
python setup.py install
```

Use for training:

```shell
python main.py -c configs/cofig_name.json
```
