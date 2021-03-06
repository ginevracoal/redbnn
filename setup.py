from distutils.core import setup

setup(
  name = 'redbnn',         # How you named your package folder (MyLib)
  packages = ['redbnn'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Reduced Bayesian Neural Networks',   # Give a short description about your library
  author = 'Ginevra Carbone',                   # Type in your name
  author_email = 'ginevracoal@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/ginevracoal/redbnn',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ginevracoal/redbnn/releases/tag/v_01',    # I explain this later on
  keywords = ['BAYESIAN', 'NEURALNETWORKS', 'TRAINING'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pyro',
          'torchvision',
      ],
)
